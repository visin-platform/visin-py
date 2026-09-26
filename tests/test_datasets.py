import io
import zipfile

import pytest
import requests
from fakes import BASE, FakeStream, ok

from visin import Datasets
from visin.cli import main
from visin.errors import ConfigurationError, TransportError, VisinError

ZOD_ID = "6aad9ac1f900dcbe46605ba9"
SIGNED = "https://files.example.test/signed/zod.zip"


def zipped(files):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return buffer.getvalue()


# Larger than one streamed chunk, so a download can be cut part-way
ARCHIVE = zipped({"zod_dataset/train.txt": "camera/1.png\n", "zod_dataset/camera/1.png": "png" * 2000})


def zod(size=None):
    size = len(ARCHIVE) if size is None else size
    return {"_id": ZOD_ID, "name": "ZOD", "archive": {"size": size, "filename": "zod_dataset.zip"}}


@pytest.fixture
def datasets(monkeypatch, client, session, uploads, tmp_path):
    """A dataset service with ZOD on it, its zip served at a signed URL."""
    monkeypatch.setenv("VISIN_DATASET_URL", BASE)
    monkeypatch.setenv("VISIN_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr("visin.datasets.HttpClient", lambda *_args, **_kwargs: client)
    signed = {"downloadUrl": SIGNED, "filename": "zod_dataset.zip"}
    session.route("GET", f"/datasets/{ZOD_ID}/download", ok(signed))
    session.route("GET", "/datasets", ok([zod()]))
    uploads.files[SIGNED] = ARCHIVE
    return Datasets()


def test_a_dataset_service_address_is_needed():
    with pytest.raises(ConfigurationError, match="VISIN_DATASET_URL"):
        Datasets()


def test_datasets_are_listed(datasets, session):
    assert [d["name"] for d in datasets.list()] == ["ZOD"]
    assert session.calls[0]["url"] == f"{BASE}/api/datasets"


def test_a_dataset_is_found_by_name_or_id(datasets, session):
    session.route("GET", f"/datasets/{ZOD_ID}", ok(zod()))
    assert datasets.get("zod")["_id"] == ZOD_ID
    assert datasets.get(ZOD_ID)["name"] == "ZOD"


def test_an_unknown_name_lists_what_there_is(datasets):
    with pytest.raises(VisinError, match="no dataset named 'waymo'; available: ZOD"):
        datasets.get("waymo")


def test_download_unpacks_and_returns_the_datasets_folder(datasets, uploads, tmp_path):
    root = datasets.download("zod")
    assert root == tmp_path / "data" / f"zod-{ZOD_ID}" / "zod_dataset"
    assert (root / "train.txt").read_text() == "camera/1.png\n"
    assert not list((tmp_path / "data").glob("*.zip*"))  # the zip is gone once unpacked
    assert len(uploads.gets) == 1


def test_a_downloaded_dataset_is_not_downloaded_again(datasets, session, uploads):
    session.route("GET", "/datasets", ok([zod()]))  # listed again for the second call
    first = datasets.download("zod")
    assert datasets.download("ZOD") == first
    assert len(uploads.gets) == 1


def test_an_interrupted_download_resumes(datasets, uploads, tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / f"{ZOD_ID}-zod_dataset.zip.part").write_bytes(ARCHIVE[:100])
    datasets.download("zod")
    assert uploads.gets[0]["headers"] == {"Range": "bytes=100-"}


def test_a_dropped_stream_is_resumed(datasets, uploads, client, sleeps):
    cut = FakeStream(200, ARCHIVE, fail_after=requests.exceptions.ChunkedEncodingError("reset"))
    uploads.get_answers.append(cut)
    datasets.download("zod")
    assert uploads.gets[1]["headers"]["Range"].startswith("bytes=")
    assert sleeps  # waited before asking again


def test_a_download_of_the_wrong_size_is_not_unpacked(datasets, session, uploads, tmp_path):
    session.routes[("GET", "/datasets")] = [ok([zod(size=len(ARCHIVE) + 10)])]
    with pytest.raises(TransportError, match="run again to resume"):
        datasets.download("zod", quiet=True)
    assert not (tmp_path / "data" / f"zod-{ZOD_ID}").exists()


def test_a_zip_that_writes_outside_its_folder_is_refused(datasets, session, uploads):
    evil = zipped({"../outside.txt": "no"})
    session.routes[("GET", "/datasets")] = [ok([zod(size=len(evil))])]
    uploads.files[SIGNED] = evil
    with pytest.raises(VisinError, match="points outside the dataset folder"):
        datasets.download("zod", quiet=True)


def test_a_dataset_without_a_zip_cannot_be_downloaded(datasets, session):
    session.routes[("GET", "/datasets")] = [ok([{"_id": ZOD_ID, "name": "ZOD"}])]
    with pytest.raises(VisinError, match="has no zip"):
        datasets.download("zod")


def test_cli_lists_and_downloads(datasets, session, capsys, tmp_path):
    session.route("GET", "/datasets", ok([zod()]))
    assert main(["datasets"]) == 0
    assert "ZOD" in capsys.readouterr().out
    assert main(["download", "zod"]) == 0
    assert capsys.readouterr().out.strip().endswith("zod_dataset")


def test_cli_names_an_unknown_dataset(datasets, capsys):
    assert main(["download", "waymo"]) == 1
    assert "no dataset named 'waymo'" in capsys.readouterr().err
