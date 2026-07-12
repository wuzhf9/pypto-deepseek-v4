import pytest

from serving.checkpoint import validate_checkpoint_directory


def test_validate_checkpoint_directory_requires_directory(tmp_path) -> None:
    missing = tmp_path / "missing"

    with pytest.raises(FileNotFoundError, match="Checkpoint directory does not exist"):
        validate_checkpoint_directory(missing)


@pytest.mark.parametrize(
    "present,missing",
    [
        (("tokenizer.json",), "model.safetensors.index.json"),
        (("model.safetensors.index.json",), "tokenizer.json"),
        ((), "tokenizer.json, model.safetensors.index.json"),
    ],
)
def test_validate_checkpoint_directory_reports_missing_required_files(tmp_path, present, missing) -> None:
    for name in present:
        (tmp_path / name).write_text("{}", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match=missing):
        validate_checkpoint_directory(tmp_path)


def test_validate_checkpoint_directory_accepts_canonical_layout(tmp_path) -> None:
    (tmp_path / "tokenizer.json").write_text("{}", encoding="utf-8")
    (tmp_path / "model.safetensors.index.json").write_text("{}", encoding="utf-8")

    assert validate_checkpoint_directory(tmp_path) == tmp_path
