"""Hugging Face Hub transfers, behind the ``visin[hf]`` extra.

Visin keeps a pointer to what lives on the Hub, never a copy, so both directions
happen on the machine that has the bytes: a dataset is downloaded here, a
checkpoint is uploaded from here. ``huggingface_hub`` reads the user's own token
(``HF_TOKEN`` or ``huggingface-cli login``); this package never sees it.

``huggingface_hub`` is imported lazily, so a plain ``pip install visin`` stays
light and a missing extra is an error that says how to fix it.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path
from typing import Any

from ..errors import ConfigurationError, VisinError

logger = logging.getLogger("visin")

EXTRA_HINT = "Hugging Face support is an extra: pip install 'visin[hf]'"


def _hub() -> Any:
    try:
        import huggingface_hub
    except ImportError as exc:
        raise ConfigurationError(EXTRA_HINT) from exc
    return huggingface_hub


def short(revision: str) -> str:
    """Git's usual short form of a commit hash."""
    return revision[:7]


def download_dataset(repo: str, revision: str, target: Path) -> None:
    """Fetch ``repo`` at the exact commit ``revision`` into ``target``.

    An interrupted download left in ``target`` is resumed by the Hub client.
    Raises :class:`~visin.errors.VisinError` for anything the Hub refuses or
    cannot be reached for, so a caller can fall back to another copy.
    """
    hub = _hub()
    try:
        hub.snapshot_download(repo_id=repo, repo_type="dataset", revision=revision, local_dir=str(target))
    except Exception as exc:
        raise VisinError(
            f"could not download {repo}@{short(revision)} from Hugging Face: {exc}. "
            "A private dataset needs your own HF_TOKEN."
        ) from exc
    shutil.rmtree(target / ".cache", ignore_errors=True)


def _tensors(path: Path) -> dict[str, Any] | None:
    """The weights of a PyTorch checkpoint when it holds a plain state dict, else ``None``.

    Reads ``path`` itself, or its ``model_state_dict`` / ``state_dict`` entry, and only when every value is a
    tensor. ``torch`` is not a dependency of this package: without it, or for a checkpoint of another shape,
    there is nothing to convert.
    """
    try:
        import torch
    except ImportError:
        return None
    state = torch.load(path, map_location="cpu", weights_only=True)
    for candidate in (
        state,
        *(state.get(key) for key in ("model_state_dict", "state_dict") if isinstance(state, dict)),
    ):
        if (
            isinstance(candidate, dict)
            and candidate
            and all(isinstance(value, torch.Tensor) for value in candidate.values())
        ):
            return {name: value.contiguous() for name, value in candidate.items()}
    return None


def to_safetensors(path: Path, directory: Path) -> Path | None:
    """Write ``model.safetensors`` for the checkpoint ``path`` into ``directory``, or return ``None``.

    ``None`` means it could not be converted (no ``torch`` or ``safetensors``, not a plain state dict, shared
    tensors): the original is uploaded alone. The format keeps tensors only, so metadata the checkpoint
    carried beside the weights is not in the converted file, which is why it is uploaded *beside* the
    original.
    """
    try:
        from safetensors.torch import save_file

        weights = _tensors(path)
        if weights is None:
            return None
        target = directory / "model.safetensors"
        save_file(weights, str(target))
    except Exception as exc:
        logger.info("not converting %s to safetensors: %s", path.name, exc)
        return None
    return target


def upload_model(
    path: Path,
    repo: str,
    *,
    path_in_repo: str | None = None,
    private: bool = True,
    card: str | None = None,
    safetensors: bool = False,
) -> str:
    """Upload a checkpoint file or folder to the model repo ``repo`` and return the commit hash.

    The repo is created when it does not exist, private unless ``private`` is
    false: unpublished work should not become public by default. ``card`` is a
    README added in a second commit, only when the repo has none, so a card
    someone wrote by hand is never overwritten. With ``safetensors``, a
    checkpoint file that is a plain state dict is also uploaded as
    ``model.safetensors`` beside the original. The returned commit is the
    last one, which holds everything.
    """
    hub = _hub()
    try:
        api = hub.HfApi()
        api.create_repo(repo_id=repo, repo_type="model", private=private, exist_ok=True)
        if path.is_dir():
            info = api.upload_folder(
                repo_id=repo, repo_type="model", folder_path=str(path), path_in_repo=path_in_repo
            )
        else:
            info = api.upload_file(
                repo_id=repo,
                repo_type="model",
                path_or_fileobj=str(path),
                path_in_repo=path_in_repo or path.name,
            )
        if safetensors and path.is_file():
            with tempfile.TemporaryDirectory() as scratch:
                converted = to_safetensors(path, Path(scratch))
                if converted is not None:
                    beside = (path_in_repo or path.name).rpartition("/")[0]
                    info = api.upload_file(
                        repo_id=repo,
                        repo_type="model",
                        path_or_fileobj=str(converted),
                        path_in_repo=f"{beside}/model.safetensors" if beside else "model.safetensors",
                    )
        if card:
            try:
                if not api.file_exists(repo_id=repo, filename="README.md", repo_type="model"):
                    info = api.upload_file(
                        repo_id=repo,
                        repo_type="model",
                        path_or_fileobj=card.encode(),
                        path_in_repo="README.md",
                        commit_message="Add model card from Visin",
                    )
            except Exception as exc:
                logger.warning(
                    "visin: model uploaded to %s, but its model card could not be added: %s", repo, exc
                )
    except Exception as exc:
        raise VisinError(f"could not upload {path} to Hugging Face repo {repo}: {exc}") from exc
    return str(info.oid)


def upload_dataset(folder: Path, repo: str, *, private: bool = True, ignore: list[str] | None = None) -> str:
    """Upload ``folder`` to the dataset repo ``repo`` and return the commit hash it made.

    The repo is created when it does not exist, private unless ``private`` is
    false: a dataset that was only ever on a lab's own server should not become
    public by default. ``ignore`` names files to leave out.
    """
    hub = _hub()
    try:
        api = hub.HfApi()
        api.create_repo(repo_id=repo, repo_type="dataset", private=private, exist_ok=True)
        info = api.upload_folder(
            repo_id=repo, repo_type="dataset", folder_path=str(folder), ignore_patterns=ignore
        )
    except Exception as exc:
        raise VisinError(f"could not upload {folder} to Hugging Face dataset repo {repo}: {exc}") from exc
    return str(info.oid)
