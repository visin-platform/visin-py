import io
import json
import sys
import zipfile

import pytest
import requests
from fakes import BASE, FakeStream, ok, refused

from visin import Datasets
from visin.cli import main
from visin.datasets import MARKER
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
    session.route("GET", f"/datasets/{ZOD_ID}/download", ok(signed), ok(signed))
    session.route("GET", "/datasets", ok([zod()]))
    uploads.files[SIGNED] = ARCHIVE
    return Datasets()


def test_a_dataset_service_address_is_needed():
    with pytest.raises(ConfigurationError, match=r"VISIN_DATASET_URL.*dataset-api\.visin\.eu"):
        Datasets()


def test_datasets_are_listed(datasets, session):
    assert [d.name for d in datasets.list()] == ["ZOD"]
    assert session.calls[0]["url"] == f"{BASE}/api/datasets"


def test_a_dataset_is_found_by_name_or_id(datasets, session):
    session.route("GET", f"/datasets/{ZOD_ID}", ok(zod()))
    assert datasets.get("zod").id == ZOD_ID
    assert datasets.get(ZOD_ID).name == "ZOD"


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


def test_an_interrupted_download_resumes(datasets, uploads, tmp_path, monkeypatch):
    stream = Terminal()
    monkeypatch.setattr("visin.datasets.sys.stderr", stream)
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / f"{ZOD_ID}-zod_dataset.zip.part").write_bytes(ARCHIVE[:100])
    datasets.download("zod")
    assert uploads.gets[0]["headers"] == {"Range": "bytes=100-"}
    assert f"{100 / len(ARCHIVE):6.1%}" in stream.getvalue()
    assert "100 B /" in stream.getvalue()


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


def test_every_page_of_datasets_is_listed(datasets, session, monkeypatch):
    monkeypatch.setattr("visin.datasets.PAGE_SIZE", 1)
    other = {"_id": "7bbd9ac1f900dcbe46605bb0", "name": "Other"}
    pages = {"pagination": {"pages": 2}}
    session.routes[("GET", "/datasets")] = [
        ok({"datasets": [zod()], **pages}),
        ok({"datasets": [other], **pages}),
    ]
    assert [d.name for d in datasets.list()] == ["ZOD", "Other"]


def test_a_server_ignoring_page_does_not_loop(datasets, session, monkeypatch):
    monkeypatch.setattr("visin.datasets.PAGE_SIZE", 1)
    session.routes[("GET", "/datasets")] = [ok([zod()]), ok([zod()])]
    assert len(datasets.list()) == 1


class Terminal(io.StringIO):
    def isatty(self):
        return True


def test_terminal_progress_updates_one_line_with_resumed_bytes_speed_and_eta(monkeypatch):
    from visin.datasets import _Progress

    stream = Terminal()
    clock = [0.0]
    monkeypatch.setattr("visin.datasets.sys.stderr", stream)
    monkeypatch.setattr("visin.datasets.time.monotonic", lambda: clock[0])
    progress = _Progress(1000)
    progress(500, 1000)
    assert "50.0%" in stream.getvalue()
    assert "500 B / 1000 B" in stream.getvalue()
    clock[0] = 1.0
    progress(750, 1000)
    assert "75.0%" in stream.getvalue()
    assert "250 B/s" in stream.getvalue()
    assert "ETA 00:01" in stream.getvalue()
    assert "\n" not in stream.getvalue()
    progress(1000, 1000)
    progress.close()
    assert "100.0%" in stream.getvalue()
    assert stream.getvalue().count("\n") == 1


def test_terminal_progress_throttles_updates_but_always_shows_completion(monkeypatch):
    from visin.datasets import _Progress

    stream = Terminal()
    monkeypatch.setattr("visin.datasets.sys.stderr", stream)
    monkeypatch.setattr("visin.datasets.time.monotonic", lambda: 0.0)
    progress = _Progress(100)
    progress(0, 100)
    initial = stream.getvalue()
    progress(50, 100)
    assert stream.getvalue() == initial
    progress(100, 100)
    assert "100.0%" in stream.getvalue()


def test_unknown_size_shows_downloaded_bytes(monkeypatch):
    from visin.datasets import _Progress

    stream = Terminal()
    monkeypatch.setattr("visin.datasets.sys.stderr", stream)
    progress = _Progress(None)
    progress(2**20, None)
    progress.close()
    assert "downloaded 1.00 MiB" in stream.getvalue()
    assert "%" not in stream.getvalue()


def test_failed_download_ends_progress_line_without_claiming_completion(datasets, monkeypatch):
    stream = Terminal()
    monkeypatch.setattr("visin.datasets.sys.stderr", stream)

    def interrupted(*_args, **kwargs):
        kwargs["progress"](100, len(ARCHIVE))
        raise TransportError("interrupted")

    monkeypatch.setattr(datasets._client, "download_file", interrupted)
    with pytest.raises(TransportError, match="interrupted"):
        datasets.download("zod")
    assert stream.getvalue().endswith("\n")
    assert "100.0%" not in stream.getvalue()


def test_quiet_download_has_no_terminal_progress(datasets, monkeypatch):
    stream = Terminal()
    monkeypatch.setattr("visin.datasets.sys.stderr", stream)
    datasets.download("zod", quiet=True)
    assert stream.getvalue() == ""


def test_progress_fits_an_eighty_column_terminal(monkeypatch):
    from os import terminal_size

    from visin.datasets import _Progress

    stream = Terminal()
    clock = [0.0]
    monkeypatch.setattr("visin.datasets.sys.stderr", stream)
    monkeypatch.setattr("visin.datasets.time.monotonic", lambda: clock[0])
    monkeypatch.setattr("visin.datasets.shutil.get_terminal_size", lambda: terminal_size((80, 24)))
    progress = _Progress(10 * 2**30)
    progress(0, None)
    clock[0] = 1.0
    progress(2**30, None)
    line = stream.getvalue().split("\r")[-1]
    assert len(line) < 80
    assert "10.0%" in line and "1.00 GiB / 10.00 GiB" in line
    assert "ETA" in line


@pytest.mark.parametrize(
    ("unzip", "keep_archive"), [(True, False), (True, True), (False, False), (False, True)]
)
def test_download_options_control_extraction_retention_and_return_path(
    datasets, tmp_path, unzip, keep_archive
):
    result = datasets.download("zod", unzip=unzip, keep_archive=keep_archive)
    archive = tmp_path / "data" / f"{ZOD_ID}-zod_dataset.zip"
    extracted = tmp_path / "data" / f"zod-{ZOD_ID}"
    assert archive.exists() == (keep_archive or not unzip)
    assert extracted.exists() == unzip
    if unzip:
        assert (result / "train.txt").is_file()
    else:
        assert result == archive
        assert result.read_bytes() == ARCHIVE


def test_a_zip_only_download_can_later_be_extracted_without_downloading_again(datasets, session, uploads):
    archive = datasets.download("zod", unzip=False)
    session.route("GET", "/datasets", ok([zod()]))
    result = datasets.download("zod")
    assert (result / "train.txt").is_file()
    assert not archive.exists()
    assert len(uploads.gets) == 1


def test_a_repeated_zip_only_download_reuses_the_archive(datasets, session, uploads):
    first = datasets.download("zod", unzip=False)
    session.route("GET", "/datasets", ok([zod()]))
    assert datasets.download("zod", unzip=False) == first
    assert len(uploads.gets) == 1


def test_keep_archive_fetches_a_missing_zip_without_extracting_again(datasets, session, uploads, monkeypatch):
    first = datasets.download("zod")
    session.route("GET", "/datasets", ok([zod()]))

    def unexpected_extraction(*_args):
        pytest.fail("completed dataset should not be extracted again")

    monkeypatch.setattr("visin.datasets._unzip", unexpected_extraction)
    assert datasets.download("zod", keep_archive=True) == first
    assert len(uploads.gets) == 2
    assert list(datasets.directory.glob("*.zip"))


def test_failed_extraction_reuses_the_completed_archive_on_retry(datasets, session, uploads, monkeypatch):
    from visin.datasets import _unzip

    def interrupted(*_args):
        raise OSError("disk full")

    monkeypatch.setattr("visin.datasets._unzip", interrupted)
    with pytest.raises(OSError, match="disk full"):
        datasets.download("zod")
    assert list(datasets.directory.glob("*.zip"))
    session.route("GET", "/datasets", ok([zod()]))
    monkeypatch.setattr("visin.datasets._unzip", _unzip)
    assert (datasets.download("zod") / "train.txt").is_file()
    assert len(uploads.gets) == 1


def test_an_archive_with_the_wrong_size_is_downloaded_again(datasets, tmp_path, uploads):
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / f"{ZOD_ID}-zod_dataset.zip").write_bytes(b"incomplete")
    result = datasets.download("zod", unzip=False)
    assert result.read_bytes() == ARCHIVE
    assert len(uploads.gets) == 1


def test_archive_cleanup_failure_does_not_fail_a_completed_dataset(datasets, monkeypatch, caplog):
    from pathlib import Path

    unlink = Path.unlink

    def fail_zip_delete(path, *args, **kwargs):
        if path.suffix == ".zip":
            raise PermissionError("archive is read-only")
        return unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_zip_delete)
    assert (datasets.download("zod") / "train.txt").is_file()
    assert "could not delete archive" in caplog.text


@pytest.mark.parametrize(("flags", "zip_only"), [(["--no-unzip"], True), (["--keep-archive"], False)])
def test_cli_download_options(datasets, capsys, flags, zip_only):
    assert main(["download", "zod", *flags]) == 0
    output = capsys.readouterr().out.strip()
    assert output.endswith(".zip") == zip_only
    assert list(datasets.directory.glob("*.zip"))
    assert bool(list(datasets.directory.glob(f"zod-{ZOD_ID}"))) != zip_only


def test_cached_datasets_lists_what_is_on_disk_without_a_server(datasets, tmp_path):
    datasets.download("zod", keep_archive=True)
    (tmp_path / "data" / "9f-other.zip.part").write_bytes(b"x" * 10)
    from visin import cached_datasets

    kinds = {item.kind: item for item in cached_datasets(tmp_path / "data")}
    assert set(kinds) == {"dataset", "archive", "partial"}
    assert kinds["dataset"].name == "ZOD" and kinds["dataset"].id == ZOD_ID
    assert kinds["archive"].size == len(ARCHIVE)


def test_remove_cached_deletes_the_dataset_its_zip_and_leaves_others(datasets, tmp_path):
    datasets.download("zod", keep_archive=True)
    stranger = tmp_path / "data" / "ffff-other.zip"
    stranger.write_bytes(b"x")
    from visin import remove_cached

    removed = remove_cached("zod", tmp_path / "data")
    assert {item.kind for item in removed} == {"dataset", "archive"}
    assert stranger.exists()
    assert not list((tmp_path / "data").glob(f"*{ZOD_ID}*"))
    with pytest.raises(VisinError, match="nothing downloaded"):
        remove_cached("zod", tmp_path / "data")


def test_an_empty_or_missing_data_directory_lists_nothing(tmp_path):
    from visin import cached_datasets

    assert cached_datasets(tmp_path / "nowhere") == []


def test_remove_cached_matches_names_exactly_not_by_prefix(tmp_path):
    from visin import remove_cached

    data = tmp_path / "data"
    (data / "zod-full-aaa1.unpacking").mkdir(parents=True)
    (data / "zo-bbb2.unpacking").mkdir()
    removed = remove_cached("zo", data)
    assert [item.path.name for item in removed] == ["zo-bbb2.unpacking"]
    assert (data / "zod-full-aaa1.unpacking").exists()


def test_a_same_size_replacement_downloads_its_actual_revision(datasets, session, uploads):
    first = zod()
    first["archive"]["uploadedAt"] = "2026-01-01T00:00:00.000Z"
    session.routes[("GET", "/datasets")] = [ok([first])]
    root = datasets.download("zod", keep_archive=True)
    replacement = zipped(
        {"zod_dataset/train.txt": "camera/2.png\n", "zod_dataset/camera/1.png": "png" * 2000}
    )
    second = zod()
    second["archive"]["uploadedAt"] = "2026-02-01T00:00:00.000Z"
    session.routes[("GET", "/datasets")] = [ok([second])]
    uploads.files[SIGNED] = replacement
    assert datasets.download("zod", keep_archive=True) == root
    assert len(uploads.gets) == 2
    assert (root / "train.txt").read_text() == "camera/2.png\n"


@pytest.mark.parametrize("already_extracted", [False, True])
def test_download_pins_the_archive_signed_after_metadata_was_read(
    datasets, session, uploads, already_extracted
):
    import json

    first = zod()
    first["archive"]["uploadedAt"] = "2026-01-01T00:00:00.000Z"
    session.routes[("GET", "/datasets")] = [ok([first])]
    if already_extracted:
        datasets.download("zod", keep_archive=True)
        session.routes[("GET", "/datasets")] = [ok([first])]
    replacement = zipped({"zod_dataset/train.txt": "replacement content of another size"})
    session.routes[("GET", f"/datasets/{ZOD_ID}/download")] = [
        ok(
            {
                "downloadUrl": SIGNED,
                "filename": "zod_dataset.zip",
                "size": len(replacement),
                "revision": "2026-02-01T00:00:00.000Z",
            }
        )
    ]
    uploads.files[SIGNED] = replacement
    root = datasets.download("zod", quiet=True, keep_archive=True)
    assert (root / "train.txt").read_text() == "replacement content of another size"
    marker = json.loads((root.parent / ".visin-dataset.json").read_text())
    assert marker["revision"] == "2026-02-01T00:00:00.000Z"
    assert marker["size"] == len(replacement)


@pytest.mark.parametrize("contents", ["[]", "null", "{", "{}"])
@pytest.mark.parametrize("known_size", [False, True])
def test_a_damaged_cache_marker_is_rebuilt(datasets, session, uploads, contents, known_size):
    item = zod()
    if not known_size:
        item["archive"].pop("size")
    session.routes[("GET", "/datasets")] = [ok([item])]
    root = datasets.download("zod", quiet=True)
    (root.parent / ".visin-dataset.json").write_text(contents)
    session.route("GET", "/datasets", ok([item]))
    assert datasets.download("zod", quiet=True) == root
    assert len(uploads.gets) == 2


@pytest.mark.parametrize("filename", ["inner/../../escaped.zip", "inner\\..\\..\\escaped.zip"])
def test_the_download_filename_cannot_write_outside_the_cache(datasets, session, tmp_path, filename):
    session.routes[("GET", f"/datasets/{ZOD_ID}/download")] = [
        ok({"downloadUrl": SIGNED, "filename": filename})
    ]
    archive = datasets.download("zod", quiet=True, unzip=False)
    assert archive.parent == tmp_path / "data"
    assert archive.name == f"{ZOD_ID}-escaped.zip"
    assert not (tmp_path / "escaped.zip").exists()


HUB_COMMIT = "3f2a1c9d8e7b6a5f4e3d2c1b0a99887766554433"
NEXT_COMMIT = "b" * 40
SOURCE = {"provider": "hf", "repo": "acme/zod-png", "revision": HUB_COMMIT}


def on_hub(session, *, zip_too=False, source=SOURCE):
    """ZOD kept on the Hub, with the zip as a fallback when ``zip_too``."""
    item = {"_id": ZOD_ID, "name": "ZOD", "source": source}
    answer = {"source": source, "revision": source["revision"]}
    if zip_too:
        item["archive"] = {"size": len(ARCHIVE), "filename": "zod_dataset.zip"}
        answer |= {
            "downloadUrl": SIGNED,
            "filename": "zod_dataset.zip",
            "size": len(ARCHIVE),
            "archiveRevision": "2026-10-02T10:00:00.000Z",
        }
    session.routes[("GET", "/datasets")] = [ok([item]), ok([item]), ok([item])]
    session.routes[("GET", f"/datasets/{ZOD_ID}/download")] = [ok(answer), ok(answer), ok(answer)]


def test_a_hub_dataset_is_downloaded_at_its_commit_into_the_same_layout(datasets, session, hf, tmp_path):
    on_hub(session)
    root = datasets.download("zod")
    assert hf.downloads == [
        {
            "repo_id": "acme/zod-png",
            "repo_type": "dataset",
            "revision": HUB_COMMIT,
            "local_dir": str(tmp_path / "data" / f"zod-{ZOD_ID}.unpacking"),
        }
    ]
    assert (root / "images" / "1.png").read_bytes() == b"png"
    assert not (root / ".cache").exists()
    marker = json.loads((root / MARKER).read_text())
    assert marker["revision"] == HUB_COMMIT and marker["source"] == SOURCE and marker["size"] is None


def test_a_hub_dataset_is_not_downloaded_again_until_its_commit_changes(datasets, session, hf):
    on_hub(session)
    first = datasets.download("zod")
    datasets.download("zod")
    assert len(hf.downloads) == 1
    on_hub(session, source={**SOURCE, "revision": NEXT_COMMIT})
    assert (datasets.download("zod") / "README.md").read_text() == NEXT_COMMIT
    assert len(hf.downloads) == 2 and first.exists()


def test_a_cached_hub_dataset_needs_no_hub_and_no_extra(datasets, session, hf, monkeypatch):
    on_hub(session)
    datasets.download("zod")
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    session.calls.clear()
    assert datasets.download("zod").exists()
    assert not [call for call in session.calls if call["url"].endswith("/download")]


def test_the_hub_is_preferred_over_the_zip_kept_on_visin(datasets, session, uploads, hf):
    on_hub(session, zip_too=True)
    datasets.download("zod")
    assert len(hf.downloads) == 1
    assert not uploads.gets


def test_a_hub_failure_falls_back_to_the_zip_and_records_its_version(datasets, session, hf, caplog):
    on_hub(session, zip_too=True)
    hf.failure = RuntimeError("401 gated")
    root = datasets.download("zod")
    assert (root / "camera" / "1.png").exists()
    assert "using the zip kept on Visin instead" in caplog.text
    assert json.loads((root.parent / MARKER).read_text())["revision"] == "2026-10-02T10:00:00.000Z"


def test_a_hub_failure_with_no_zip_says_what_went_wrong(datasets, session, hf):
    on_hub(session)
    hf.failure = RuntimeError("401 gated")
    with pytest.raises(VisinError, match=r"could not download acme/zod-png@3f2a1c9.*HF_TOKEN"):
        datasets.download("zod")


def test_a_missing_hub_extra_is_named_and_the_zip_is_used_when_there_is_one(datasets, session, monkeypatch):
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    on_hub(session)
    with pytest.raises(ConfigurationError, match=r"visin\[hf\]"):
        datasets.download("zod")
    on_hub(session, zip_too=True)
    assert datasets.download("zod").exists()


def test_a_hub_only_dataset_has_no_zip_to_keep(datasets, session, hf):
    on_hub(session)
    with pytest.raises(VisinError, match="on Hugging Face only"):
        datasets.download("zod", unzip=False)
    with pytest.raises(VisinError, match="on Hugging Face only"):
        datasets.download("zod", keep_archive=True)
    assert not hf.downloads


def test_a_dataset_with_a_hub_source_is_downloadable_without_a_zip():
    from visin.models import Dataset

    assert Dataset.from_json({"_id": ZOD_ID, "name": "ZOD", "source": SOURCE}).downloadable
    assert Dataset.from_json({"_id": ZOD_ID, "name": "ZOD", "source": SOURCE}).source == SOURCE
    assert not Dataset.from_json({"_id": ZOD_ID, "name": "ZOD", "source": {"repo": "x"}}).downloadable


def test_a_visin_dataset_is_published_to_the_hub_and_visin_is_pointed_at_it(datasets, session, hf, uploads):
    session.route("PATCH", f"/datasets/{ZOD_ID}", ok({}))
    assert datasets.push("zod", "acme/zod-png") == HUB_COMMIT
    assert hf.created == [
        {"repo_id": "acme/zod-png", "repo_type": "dataset", "private": True, "exist_ok": True}
    ]
    (upload,) = hf.uploads
    assert upload["kind"] == "folder" and upload["repo_type"] == "dataset"
    assert upload["ignore_patterns"] == [MARKER]
    assert "camera/1.png" in upload["files"]
    (patch,) = [call for call in session.calls if call["method"] == "PATCH"]
    assert patch["json"] == {"source": {"provider": "hf", "repo": "acme/zod-png", "revision": HUB_COMMIT}}


def test_a_public_repo_must_be_asked_for(datasets, session, hf):
    session.route("PATCH", f"/datasets/{ZOD_ID}", ok({}))
    datasets.push("zod", "acme/zod-png", private=False)
    assert hf.created[0]["private"] is False


def test_a_dataset_already_on_the_hub_or_without_a_zip_is_not_published(datasets, session, hf):
    on_hub(session)
    with pytest.raises(VisinError, match=r"already on Hugging Face \(acme/zod-png@3f2a1c9\)"):
        datasets.push("zod", "acme/other")
    session.routes[("GET", "/datasets")] = [ok([{"_id": ZOD_ID, "name": "ZOD"}])]
    with pytest.raises(VisinError, match="no zip on Visin to publish"):
        datasets.push("zod", "acme/other")
    assert not hf.uploads


def test_a_failed_upload_changes_nothing_on_visin(datasets, session, hf):
    hf.failure = RuntimeError("403")
    with pytest.raises(VisinError, match="could not upload"):
        datasets.push("zod", "acme/zod-png")
    assert not [call for call in session.calls if call["method"] == "PATCH"]


def test_a_refused_link_says_where_the_data_went_and_how_to_finish(datasets, session, hf):
    session.route("PATCH", f"/datasets/{ZOD_ID}", refused(403, "Forbidden"))
    with pytest.raises(
        VisinError, match=rf"uploaded to acme/zod-png@3f2a1c9, but Visin did not accept.*{HUB_COMMIT}"
    ):
        datasets.push("zod", "acme/zod-png")


def test_cli_push_prints_the_repo_and_commit_or_the_reason(datasets, session, hf, capsys):
    session.route("PATCH", f"/datasets/{ZOD_ID}", ok({}))
    assert main(["push", "zod", "--repo", "acme/zod-png", "--public"]) == 0
    assert capsys.readouterr().out.strip() == f"acme/zod-png@{HUB_COMMIT}"
    assert hf.created[0]["private"] is False
    hf.failure = RuntimeError("401")
    session.route("GET", "/datasets", ok([zod()]))
    assert main(["push", "zod", "--repo", "acme/zod-png"]) == 1
    assert "visin push: could not upload" in capsys.readouterr().err


def test_a_cached_zip_does_not_hide_a_new_hub_source(datasets, session, hf):
    datasets.download("zod", quiet=True)
    on_hub(session, zip_too=True)
    root = datasets.download("zod", quiet=True)
    assert (root / "README.md").read_text() == HUB_COMMIT
    assert len(hf.downloads) == 1


def test_a_hub_cache_is_bound_to_the_repo_as_well_as_the_commit(datasets, session, hf):
    on_hub(session)
    datasets.download("zod", quiet=True)
    on_hub(session, source={**SOURCE, "repo": "other/zod-png"})
    datasets.download("zod", quiet=True)
    assert [call["repo_id"] for call in hf.downloads] == ["acme/zod-png", "other/zod-png"]


@pytest.mark.parametrize("same_revision", [True, False])
def test_an_interrupted_hub_download_resumes_only_the_same_source(
    datasets, session, hf, monkeypatch, same_revision
):
    from pathlib import Path

    on_hub(session)
    download = hf.snapshot_download

    def interrupted(**kwargs):
        root = Path(kwargs["local_dir"])
        (root / "old-only.png").write_bytes(b"partial")
        raise RuntimeError("connection lost")

    monkeypatch.setattr(hf, "snapshot_download", interrupted)
    with pytest.raises(VisinError, match="connection lost"):
        datasets.download("zod", quiet=True)
    monkeypatch.setattr(hf, "snapshot_download", download)
    on_hub(session, source={**SOURCE, "revision": HUB_COMMIT if same_revision else NEXT_COMMIT})
    root = datasets.download("zod", quiet=True)
    assert (root / "old-only.png").exists() is same_revision
    assert not (root / ".cache").exists()
