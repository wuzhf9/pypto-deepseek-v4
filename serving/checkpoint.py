"""Canonical DeepSeek V4 checkpoint directory validation."""

from pathlib import Path


REQUIRED_CHECKPOINT_FILES = (
    "tokenizer.json",
    "model.safetensors.index.json",
)


def validate_checkpoint_directory(checkpoint: str | Path) -> Path:
    """Return a checkpoint directory containing all canonical runtime files."""
    path = Path(checkpoint).expanduser()
    if not path.is_dir():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {path}")
    missing = [name for name in REQUIRED_CHECKPOINT_FILES if not (path / name).is_file()]
    if missing:
        names = ", ".join(missing)
        raise FileNotFoundError(f"Checkpoint directory {path} is missing required files: {names}")
    return path


__all__ = ["REQUIRED_CHECKPOINT_FILES", "validate_checkpoint_directory"]
