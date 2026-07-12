"""Root smoke entrypoint for the device-runtime-injected DeepSeek V4 runner."""

import argparse
from typing import Any

import torch

from models.config import FLASH_CONFIG
from serving.checkpoint import validate_checkpoint_directory
from serving.device_runtime import DeviceRuntime
from serving.state import DEFAULT_MAX_SEQ_LEN


DeepSeekV4Runner: Any | None = None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DeepSeek V4 Flash PyPTO runner smoke entrypoint.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("-p", "--platform", type=str, default="a2a3")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("-s", "--seq-len", type=int, default=1)
    parser.add_argument("--max-layers", type=int, default=1)
    parser.add_argument("--enable-l2-swimlane", action="store_true", default=False)
    parser.add_argument("--keep-prefill-routed-staging", action="store_true", default=False)
    parser.add_argument("--no-head", action="store_true", default=False)
    parser.add_argument("--decode-steps", type=int, default=0)
    parser.add_argument("--profile", action="store_true", default=False)
    parser.add_argument("--verbose-layer-log", action="store_true", default=False)
    parser.add_argument("--expert-cache-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    global DeepSeekV4Runner
    args = parse_args(argv)
    checkpoint = validate_checkpoint_directory(args.checkpoint)
    if args.decode_steps < 0:
        raise ValueError(f"decode steps must be non-negative, got {args.decode_steps}")
    if args.seq_len + args.decode_steps > DEFAULT_MAX_SEQ_LEN:
        raise ValueError(
            f"seq_len + decode_steps must be <= {DEFAULT_MAX_SEQ_LEN}, "
            f"got {args.seq_len} + {args.decode_steps}"
        )

    torch.manual_seed(args.seed)
    input_ids = torch.randint(0, FLASH_CONFIG.vocab_size, (1, args.seq_len), dtype=torch.int64)
    if DeepSeekV4Runner is None:
        from serving.runner import DeepSeekV4Runner as _DeepSeekV4Runner

        DeepSeekV4Runner = _DeepSeekV4Runner
    runtime = DeviceRuntime(
        platform=args.platform,
        device_id=args.device,
        runtime_cfg={"enable_l2_swimlane": args.enable_l2_swimlane},
        keep_prefill_routed_staging=args.keep_prefill_routed_staging,
    )
    try:
        runner = DeepSeekV4Runner(
            str(checkpoint),
            runtime=runtime,
            max_layers=args.max_layers,
            run_head=not args.no_head,
            profile=args.profile,
            verbose_layer_log=args.verbose_layer_log,
            expert_cache_dir=args.expert_cache_dir,
        )
    except BaseException:
        runtime.close()
        raise

    try:
        out = runner.prefill(input_ids)
        finite = bool(torch.isfinite(out.float()).all().item())
        print(
            f"[RUNNER] prefill ok: input_ids={tuple(input_ids.shape)} "
            f"output={tuple(out.shape)} dtype={out.dtype} finite={finite}",
            flush=True,
        )
        if not finite:
            return 1

        next_ids = _next_decode_input(out, run_head=not args.no_head)
        for step in range(args.decode_steps):
            start_pos = args.seq_len + step
            out = runner.decode(next_ids, start_pos=start_pos)
            finite = bool(torch.isfinite(out.float()).all().item())
            print(
                f"[RUNNER] decode ok: step={step + 1}/{args.decode_steps} "
                f"start_pos={start_pos} input_ids={tuple(next_ids.shape)} "
                f"output={tuple(out.shape)} dtype={out.dtype} finite={finite}",
                flush=True,
            )
            if not finite:
                return 1
            next_ids = _next_decode_input(out, run_head=not args.no_head)
    finally:
        runner.close()

    return 0


def _next_decode_input(out: torch.Tensor, *, run_head: bool) -> torch.Tensor:
    if run_head:
        return torch.argmax(out, dim=-1).view(1, 1).to(dtype=torch.int64, device="cpu").contiguous()
    return torch.randint(0, FLASH_CONFIG.vocab_size, (1, 1), dtype=torch.int64)


if __name__ == "__main__":
    raise SystemExit(main())
