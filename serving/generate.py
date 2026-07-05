"""Text generation entrypoint for DeepSeek V4 Flash PyPTO inference."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib.util
from pathlib import Path
import time
from types import ModuleType
from typing import Any, Callable, Literal

import torch

from models.config import FLASH_CONFIG
from serving.state import DEFAULT_MAX_SEQ_LEN


ThinkingMode = Literal["chat", "thinking"]

AutoTokenizer: Any | None = None
DeepSeekV4Runner: Any | None = None


@dataclass(frozen=True)
class EncodingHelpers:
    encode_messages: Callable[..., str]
    eos_token: str
    parse_message_from_completion_text: Callable[[str, str], dict[str, Any]]


@dataclass(frozen=True)
class GenerationResult:
    text: str
    prompt_text: str
    prompt_tokens: int
    generated_tokens: int
    elapsed_s: float

    @property
    def output_tps(self) -> float:
        return self.generated_tokens / self.elapsed_s if self.elapsed_s > 0 else 0.0


def resolve_tokenizer_path(checkpoint_path: str | Path, tokenizer_path: str | Path | None = None) -> Path:
    if tokenizer_path is not None:
        return Path(tokenizer_path)

    checkpoint = Path(checkpoint_path)
    if (checkpoint / "tokenizer.json").exists():
        return checkpoint
    low_vram_cache = checkpoint / "bf16_lowvram_cache"
    if (low_vram_cache / "tokenizer.json").exists():
        return low_vram_cache
    raise FileNotFoundError(
        f"Could not find tokenizer.json under {checkpoint} or {low_vram_cache}; "
        "pass --tokenizer-path explicitly."
    )


def load_encoding_helpers(
    checkpoint_path: str | Path,
    *,
    encoding_path: str | Path | None = None,
) -> EncodingHelpers:
    module = _load_encoding_module(checkpoint_path, encoding_path=encoding_path)
    return EncodingHelpers(
        encode_messages=module.encode_messages,
        eos_token=module.eos_token,
        parse_message_from_completion_text=module.parse_message_from_completion_text,
    )


def encode_prompt(
    tokenizer: Any,
    prompt: str,
    *,
    thinking_mode: ThinkingMode = "chat",
    helpers: EncodingHelpers,
) -> tuple[list[int], str]:
    messages = [{"role": "user", "content": prompt}]
    prompt_text = helpers.encode_messages(messages, thinking_mode=thinking_mode)
    return tokenizer.encode(prompt_text), prompt_text


def format_completion(
    completion_text: str,
    *,
    thinking_mode: ThinkingMode,
    helpers: EncodingHelpers,
    parse_eos: bool = False,
) -> str:
    if not parse_eos or helpers.eos_token not in completion_text:
        return completion_text
    message = helpers.parse_message_from_completion_text(completion_text, thinking_mode=thinking_mode)
    return str(message["content"])


def select_next_token(logits: torch.Tensor, *, temperature: float = 0.0) -> torch.Tensor:
    if temperature <= 0:
        return logits.argmax(dim=-1)
    scaled = logits / max(temperature, 1e-5)
    probs = torch.softmax(scaled, dim=-1, dtype=torch.float32)
    return probs.div_(torch.empty_like(probs).exponential_()).argmax(dim=-1)


@torch.inference_mode()
def generate_ids(
    runner: Any,
    prompt_ids: list[int] | torch.Tensor,
    *,
    max_new_tokens: int,
    eos_id: int | None,
    temperature: float = 0.0,
    include_eos: bool = False,
    max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
) -> list[int]:
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    if max_new_tokens == 0:
        return []

    ids = _as_input_ids(prompt_ids)
    if ids.shape[1] == 0:
        raise ValueError("prompt_ids must contain at least one token")
    prompt_len = int(ids.shape[1])
    if prompt_len + max_new_tokens > max_seq_len:
        raise ValueError("prompt length plus max_new_tokens exceeds max_seq_len")

    logits = runner.prefill(ids)
    generated: list[int] = []
    cur_pos = prompt_len

    for step in range(max_new_tokens):
        next_token = select_next_token(logits, temperature=temperature)
        token_id = int(next_token.item())

        if eos_id is not None and token_id == eos_id:
            if include_eos:
                generated.append(token_id)
            break

        generated.append(token_id)
        if step + 1 == max_new_tokens:
            break

        token_ids = torch.tensor([[token_id]], dtype=torch.int64)
        logits = runner.decode(token_ids, start_pos=cur_pos)
        cur_pos += 1

    return generated


def run_generation(args: argparse.Namespace) -> GenerationResult:
    tokenizer = _load_tokenizer(resolve_tokenizer_path(args.checkpoint, args.tokenizer_path))
    helpers = load_encoding_helpers(args.checkpoint, encoding_path=args.encoding_path)
    prompt_ids, prompt_text = encode_prompt(
        tokenizer,
        args.prompt,
        thinking_mode=args.thinking_mode,
        helpers=helpers,
    )
    if args.print_prompt:
        print(prompt_text, flush=True)

    torch.manual_seed(args.seed)
    runner = _create_runner(args)
    try:
        start = time.perf_counter()
        generated = generate_ids(
            runner,
            prompt_ids,
            max_new_tokens=args.max_new_tokens,
            eos_id=tokenizer.eos_token_id,
            temperature=args.temperature,
            include_eos=args.include_eos or args.parse_eos,
            max_seq_len=args.max_seq_len,
        )
        elapsed_s = time.perf_counter() - start
    finally:
        runner.close()

    completion_text = tokenizer.decode(
        generated,
        skip_special_tokens=False if args.parse_eos else args.skip_special_tokens,
    )
    text = format_completion(
        completion_text,
        thinking_mode=args.thinking_mode,
        helpers=helpers,
        parse_eos=args.parse_eos,
    )
    return GenerationResult(
        text=text,
        prompt_text=prompt_text,
        prompt_tokens=len(prompt_ids),
        generated_tokens=len(generated),
        elapsed_s=elapsed_s,
    )


def print_stats(result: GenerationResult) -> None:
    print("[stats]")
    print(f"prompt_tokens: {result.prompt_tokens}")
    print(f"generated_tokens: {result.generated_tokens}")
    print(f"elapsed_s: {result.elapsed_s:.3f}")
    print(f"output_tps: {result.output_tps:.3f}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DeepSeek V4 Flash PyPTO text generation.")
    parser.add_argument("--checkpoint", type=str, default="../deepseek_v4_flash")
    parser.add_argument("--weight-index", type=str, default=None)
    parser.add_argument("--tokenizer-path", type=str, default=None)
    parser.add_argument("--encoding-path", type=str, default=None)
    parser.add_argument("--routed-pack-cache-dir", type=str, default=None)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--thinking-mode", choices=["chat", "thinking"], default="chat")
    parser.add_argument("--max-new-tokens", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-seq-len", type=int, default=DEFAULT_MAX_SEQ_LEN)
    parser.add_argument("--max-layers", type=int, default=FLASH_CONFIG.n_layers)
    parser.add_argument("--backend", choices=["direct", "worker"], default="direct")
    parser.add_argument("-p", "--platform", type=str, default="a2a3")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=33377335)
    parser.add_argument("--include-eos", action="store_true")
    parser.add_argument("--parse-eos", action="store_true")
    parser.add_argument("--skip-special-tokens", action="store_true")
    parser.add_argument("--print-prompt", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--stats", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_generation(args)
    print(f"AI: {result.text}")
    if args.stats:
        print_stats(result)
    return 0


def _load_encoding_module(
    checkpoint_path: str | Path,
    *,
    encoding_path: str | Path | None,
) -> ModuleType:
    candidates = _encoding_candidates(checkpoint_path, encoding_path)
    for candidate in candidates:
        module_path = candidate / "encoding_dsv4.py" if candidate.is_dir() else candidate
        if module_path.exists():
            spec = importlib.util.spec_from_file_location("dsv4_encoding_dsv4", module_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Could not load encoding module from {module_path}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not find encoding_dsv4.py; searched: {searched}")


def _encoding_candidates(
    checkpoint_path: str | Path,
    encoding_path: str | Path | None,
) -> list[Path]:
    if encoding_path is not None:
        return [Path(encoding_path)]

    checkpoint = Path(checkpoint_path)
    repo_root = Path(__file__).resolve().parents[1]
    return [
        checkpoint / "encoding",
        checkpoint,
        repo_root.parent / "deepseek_v4_flash" / "encoding",
        Path.cwd().parent / "deepseek_v4_flash" / "encoding",
    ]


def _load_tokenizer(tokenizer_path: Path) -> Any:
    global AutoTokenizer
    if AutoTokenizer is None:
        from transformers import AutoTokenizer as _AutoTokenizer

        AutoTokenizer = _AutoTokenizer
    return AutoTokenizer.from_pretrained(tokenizer_path)


def _create_runner(args: argparse.Namespace) -> Any:
    global DeepSeekV4Runner
    if DeepSeekV4Runner is None:
        from serving.runner import DeepSeekV4Runner as _DeepSeekV4Runner

        DeepSeekV4Runner = _DeepSeekV4Runner
    return DeepSeekV4Runner(
        args.checkpoint,
        weight_index=args.weight_index,
        platform=args.platform,
        device_id=args.device,
        backend=args.backend,
        max_seq_len=args.max_seq_len,
        max_layers=args.max_layers,
        run_head=True,
        profile=args.profile,
        routed_pack_cache_dir=args.routed_pack_cache_dir,
    )


def _as_input_ids(prompt_ids: list[int] | torch.Tensor) -> torch.Tensor:
    ids = torch.as_tensor(prompt_ids, dtype=torch.int64, device="cpu")
    if ids.ndim == 1:
        ids = ids.view(1, -1)
    if ids.ndim != 2 or ids.shape[0] != 1:
        raise ValueError(f"prompt_ids must have shape [S] or [1, S], got {tuple(ids.shape)}")
    return ids.contiguous()


if __name__ == "__main__":
    raise SystemExit(main())
