"""Small Python helpers shared by DeepSeek V4 Flash bf16 kernels."""


def ceil_div(value: int, divisor: int) -> int:
    if divisor <= 0:
        raise ValueError(f"divisor must be positive, got {divisor}")
    return (value + divisor - 1) // divisor


def assert_divisible(value: int, divisor: int, name: str) -> None:
    if divisor <= 0:
        raise ValueError(f"{name} divisor must be positive, got {divisor}")
    if value % divisor != 0:
        raise ValueError(f"{name} must be divisible by {divisor}, got {value}")


def token_count(batch: int, seq: int) -> int:
    return batch * seq


__all__ = [
    "ceil_div",
    "assert_divisible",
    "token_count",
]
