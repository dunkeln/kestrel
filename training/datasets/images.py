from io import BytesIO
from pathlib import Path
import tarfile
from typing import Any
import zipfile

from PIL import Image

from training.datasets.contracts import EvalSample
from training.datasets.loaders import DATASET_CARDS

DEFAULT_DATASET_ROOT = Path("artifacts/datasets")


class MissingSampleImageError(FileNotFoundError):
    pass


def resolve_sample_image(
    sample: EvalSample,
    dataset_root: Path = DEFAULT_DATASET_ROOT,
) -> Any:
    if not isinstance(sample.image, str):
        return sample.image

    image_ref = str(sample.metadata.get("image_ref") or sample.image)
    direct_path = _resolve_extracted_path(sample, image_ref, dataset_root)
    if direct_path is not None:
        return str(direct_path)

    if "image_ref" not in sample.metadata and "image_archive" not in sample.metadata:
        return sample.image

    archive_ref = sample.metadata.get("image_archive")
    if archive_ref:
        archive_path = _resolve_archive(sample, str(archive_ref), dataset_root)
        if archive_path is not None:
            return _read_archive_image(archive_path, image_ref)

    raise _missing_image_error(sample, image_ref, archive_ref, dataset_root)


def _resolve_extracted_path(
    sample: EvalSample,
    image_ref: str,
    dataset_root: Path,
) -> Path | None:
    candidate = Path(image_ref)
    if candidate.exists():
        return candidate

    normalized_ref = image_ref.removeprefix("./")
    candidates = [
        dataset_root / sample.dataset / normalized_ref,
        dataset_root / normalized_ref,
        dataset_root / sample.dataset / candidate.name,
        dataset_root / candidate.name,
    ]
    return next((path for path in candidates if path.exists()), None)


def _resolve_archive(
    sample: EvalSample,
    archive_ref: str,
    dataset_root: Path,
) -> Path | None:
    candidate = Path(archive_ref)
    if candidate.exists():
        return candidate

    archive_path = dataset_root / sample.dataset / archive_ref
    if archive_path.exists():
        return archive_path

    return _download_archive(sample, archive_ref, dataset_root)


def _download_archive(
    sample: EvalSample,
    archive_ref: str,
    dataset_root: Path,
) -> Path | None:
    dataset_card = DATASET_CARDS.get(sample.dataset)
    if dataset_card is None:
        return None

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        return None

    try:
        downloaded = hf_hub_download(
            repo_id=dataset_card["path"],
            filename=archive_ref,
            repo_type="dataset",
            local_dir=dataset_root / sample.dataset,
        )
    except Exception as error:
        raise MissingSampleImageError(
            f"Could not download image archive `{archive_ref}` for dataset "
            f"`{sample.dataset}` from `{dataset_card['path']}`: {error}"
        ) from error

    return Path(downloaded)


def _read_archive_image(archive_path: Path, image_ref: str) -> Image.Image:
    suffixes = "".join(archive_path.suffixes)
    if archive_path.suffix == ".zip":
        return _read_zip_image(archive_path, image_ref)
    if suffixes.endswith(".tar.gz") or archive_path.suffix == ".tar":
        return _read_tar_image(archive_path, image_ref)
    raise MissingSampleImageError(f"Unsupported image archive: `{archive_path}`.")


def _read_zip_image(archive_path: Path, image_ref: str) -> Image.Image:
    with zipfile.ZipFile(archive_path) as archive:
        member = _find_archive_member(archive.namelist(), image_ref)
        with archive.open(member) as handle:
            return _load_image(handle.read())


def _read_tar_image(archive_path: Path, image_ref: str) -> Image.Image:
    with tarfile.open(archive_path) as archive:
        names = archive.getnames()
        member = _find_archive_member(names, image_ref)
        extracted = archive.extractfile(member)
        if extracted is None:
            raise MissingSampleImageError(
                f"Could not read `{member}` from `{archive_path}`."
            )
        return _load_image(extracted.read())


def _find_archive_member(names: list[str], image_ref: str) -> str:
    normalized = image_ref.removeprefix("./")
    candidates = list(
        dict.fromkeys(
            candidate
            for candidate in [
                normalized,
                "/".join(Path(normalized).parts[1:]),
                Path(normalized).name,
            ]
            if candidate
        )
    )
    for candidate in candidates:
        if candidate in names:
            return candidate

    suffix_matches = [
        name for name in names if any(name.endswith(candidate) for candidate in candidates)
    ]
    if len(suffix_matches) == 1:
        return suffix_matches[0]

    raise MissingSampleImageError(f"Could not find `{image_ref}` in image archive.")


def _load_image(payload: bytes) -> Image.Image:
    image = Image.open(BytesIO(payload))
    image.load()
    return image


def _missing_image_error(
    sample: EvalSample,
    image_ref: str,
    archive_ref: Any,
    dataset_root: Path,
) -> MissingSampleImageError:
    archive_text = ""
    if archive_ref:
        archive_text = (
            f" Expected archive `{archive_ref}` under "
            f"`{dataset_root / sample.dataset}`."
        )
    return MissingSampleImageError(
        f"Missing image for sample `{sample.id}`: `{image_ref}`.{archive_text}"
    )
