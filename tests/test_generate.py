from types import SimpleNamespace

import pytest
import torch

from serving import generate


class FakeTokenizer:
    eos_token_id = 9

    def __init__(self):
        self.encoded_text = None
        self.decoded_ids = None
        self.skip_special_tokens = None

    def encode(self, text):
        self.encoded_text = text
        return [1, 2, 3]

    def decode(self, token_ids, skip_special_tokens=False):
        self.decoded_ids = list(token_ids)
        self.skip_special_tokens = skip_special_tokens
        if list(token_ids) == [4, self.eos_token_id]:
            return "done<eos>"
        return ",".join(str(token_id) for token_id in token_ids)


class FakeRunner:
    def __init__(self, logits):
        self.logits = [torch.tensor(item, dtype=torch.float32) for item in logits]
        self.prefill_calls = []
        self.decode_calls = []
        self.closed = False

    def prefill(self, input_ids):
        self.prefill_calls.append(input_ids.clone())
        return self.logits.pop(0)

    def decode(self, input_ids, *, start_pos):
        self.decode_calls.append((input_ids.clone(), start_pos))
        return self.logits.pop(0)

    def close(self):
        self.closed = True


def _helpers():
    def encode_messages(messages, thinking_mode):
        assert thinking_mode in {"chat", "thinking"}
        return f"<bos><user>{messages[0]['content']}</user><assistant></think>"

    def parse_message_from_completion_text(text, thinking_mode):
        assert thinking_mode == "chat"
        return {"content": text.removesuffix("<eos>")}

    return generate.EncodingHelpers(
        encode_messages=encode_messages,
        eos_token="<eos>",
        parse_message_from_completion_text=parse_message_from_completion_text,
    )


def test_encode_prompt_uses_deepseek_v4_message_encoder():
    tokenizer = FakeTokenizer()

    prompt_ids, prompt_text = generate.encode_prompt(
        tokenizer,
        "hello",
        thinking_mode="chat",
        helpers=_helpers(),
    )

    assert prompt_ids == [1, 2, 3]
    assert tokenizer.encoded_text == prompt_text
    assert "hello" in prompt_text
    assert prompt_text.endswith("</think>")


def test_resolve_prompt_text_preserves_utf8_file_contents(tmp_path):
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("第一行\n第二行\n", encoding="utf-8")

    assert generate.resolve_prompt_text(None, prompt_file) == "第一行\n第二行\n"
    assert generate.resolve_prompt_text("literal prompt", None) == "literal prompt"


def test_resolve_prompt_text_reports_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        generate.resolve_prompt_text(None, tmp_path / "missing.txt")
    with pytest.raises(ValueError, match="one of"):
        generate.resolve_prompt_text(None, None)
    assert generate.resolve_prompt_text("literal", tmp_path / "missing.txt") == "literal"


def test_parse_args_requires_a_prompt_source_and_prefers_literal_prompt(tmp_path, capsys):
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("from file", encoding="utf-8")

    literal_args = generate.parse_args(["--prompt", "literal"])
    file_args = generate.parse_args(["--prompt-file", str(prompt_file)])
    worker_args = generate.parse_args(["--prompt", "literal", "--backend", "worker"])

    assert literal_args.prompt == "literal"
    assert literal_args.prompt_file is None
    assert literal_args.backend == "worker"
    assert file_args.prompt is None
    assert file_args.prompt_file == prompt_file
    assert worker_args.backend == "worker"
    with pytest.raises(SystemExit):
        generate.parse_args([])
    with pytest.raises(SystemExit):
        generate.parse_args(["--prompt", "literal", "--backend", "direct"])
    both_args = generate.parse_args(["--prompt", "literal", "--prompt-file", str(prompt_file)])
    assert both_args.prompt == "literal"
    assert both_args.prompt_file == prompt_file
    assert "--prompt takes precedence" in capsys.readouterr().err


def test_format_completion_parses_eos_terminated_chat_response():
    helpers = _helpers()

    assert generate.format_completion("done<eos>", thinking_mode="chat", helpers=helpers, parse_eos=True) == "done"
    assert generate.format_completion("done<eos>", thinking_mode="chat", helpers=helpers, parse_eos=False) == "done<eos>"


def test_select_next_token_matches_greedy_official_path():
    logits = torch.tensor([[1.0, 3.0, 2.0]], dtype=torch.float32)

    assert int(generate.select_next_token(logits, temperature=0.0).item()) == 1


def test_generate_ids_prefill_then_decode_until_max_tokens():
    runner = FakeRunner(
        [
            [[0.0, 0.0, 5.0, 0.0]],
            [[0.0, 0.0, 0.0, 7.0]],
            [[0.0, 6.0, 0.0, 0.0]],
        ]
    )

    generated = generate.generate_ids(
        runner,
        [7, 8],
        max_new_tokens=3,
        eos_id=9,
        temperature=0.0,
        max_seq_len=8,
    )

    assert generated == [2, 3, 1]
    assert torch.equal(runner.prefill_calls[0], torch.tensor([[7, 8]], dtype=torch.int64))
    assert torch.equal(runner.decode_calls[0][0], torch.tensor([[2]], dtype=torch.int64))
    assert runner.decode_calls[0][1] == 2
    assert torch.equal(runner.decode_calls[1][0], torch.tensor([[3]], dtype=torch.int64))
    assert runner.decode_calls[1][1] == 3


def test_generate_ids_stops_on_eos_without_extra_decode():
    runner = FakeRunner([[[0.0, 0.0, 0.0, 9.0]]])

    generated = generate.generate_ids(
        runner,
        [7, 8],
        max_new_tokens=3,
        eos_id=3,
        temperature=0.0,
        include_eos=False,
    )

    assert generated == []
    assert runner.decode_calls == []


def test_generate_ids_can_include_eos():
    runner = FakeRunner([[[0.0, 0.0, 0.0, 9.0]]])

    generated = generate.generate_ids(
        runner,
        [7, 8],
        max_new_tokens=3,
        eos_id=3,
        temperature=0.0,
        include_eos=True,
    )

    assert generated == [3]


def test_generate_ids_validates_context_length():
    with pytest.raises(ValueError, match="exceeds max_seq_len"):
        generate.generate_ids(
            FakeRunner([[[1.0]]]),
            [1, 2, 3],
            max_new_tokens=2,
            eos_id=None,
            max_seq_len=4,
        )


def test_run_generation_wires_tokenizer_encoding_runner_and_decode(monkeypatch, tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "tokenizer.json").write_text("{}", encoding="utf-8")

    fake_tokenizer = FakeTokenizer()
    fake_runner = FakeRunner(
        [
            [[0.0, 0.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 7.0]],
        ]
    )
    fake_backend = object()
    captured = {}

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(path):
            captured["tokenizer_path"] = path
            return fake_tokenizer

    class FakeRunnerClass:
        def __init__(self, *args, **kwargs):
            captured["runner_args"] = args
            captured["runner_kwargs"] = kwargs

        def prefill(self, input_ids):
            return fake_runner.prefill(input_ids)

        def decode(self, input_ids, *, start_pos):
            return fake_runner.decode(input_ids, start_pos=start_pos)

        def close(self):
            fake_runner.close()

    def fake_create_backend(name, *, platform, device_id, runtime_cfg, keep_prefill_routed_staging):
        captured["backend_factory"] = (
            name,
            platform,
            device_id,
            runtime_cfg,
            keep_prefill_routed_staging,
        )
        return fake_backend

    monkeypatch.setattr(generate, "AutoTokenizer", FakeAutoTokenizer)
    monkeypatch.setattr(generate, "DeepSeekV4Runner", FakeRunnerClass)
    monkeypatch.setattr(generate, "create_backend", fake_create_backend)
    monkeypatch.setattr(generate, "load_encoding_helpers", lambda *_args, **_kwargs: _helpers())

    args = SimpleNamespace(
        checkpoint=str(checkpoint),
        weight_index=None,
        tokenizer_path=None,
        encoding_path=None,
        expert_cache_dir="cache",
        prompt="hello",
        prompt_file=None,
        thinking_mode="chat",
        max_new_tokens=2,
        temperature=0.0,
        max_seq_len=16,
        max_layers=43,
        backend="worker",
        platform="a2a3",
        device=0,
        seed=1,
        include_eos=False,
        parse_eos=False,
        skip_special_tokens=True,
        print_prompt=False,
        profile=False,
        verbose_layer_log=False,
    )

    result = generate.run_generation(args)

    assert captured["tokenizer_path"] == checkpoint
    assert captured["runner_args"] == (str(checkpoint),)
    assert captured["backend_factory"] == (
        "worker",
        "a2a3",
        0,
        {"enable_l2_swimlane": False},
        False,
    )
    assert captured["runner_kwargs"]["backend"] is fake_backend
    assert "platform" not in captured["runner_kwargs"]
    assert "device_id" not in captured["runner_kwargs"]
    assert captured["runner_kwargs"]["expert_cache_dir"] == "cache"
    assert captured["runner_kwargs"]["verbose_layer_log"] is False
    assert result.text == "4"
    assert result.prompt == "hello"
    assert result.prompt_tokens == 3
    assert result.generated_tokens == 1
    assert fake_tokenizer.skip_special_tokens is True
    assert fake_runner.closed is True


def test_create_runner_closes_backend_when_runner_initialization_fails(monkeypatch):
    class FakeBackend:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    backend = FakeBackend()

    class FailingRunner:
        def __init__(self, *_args, **_kwargs):
            raise RuntimeError("runner init failed")

    monkeypatch.setattr(generate, "create_backend", lambda *_args, **_kwargs: backend)
    monkeypatch.setattr(generate, "DeepSeekV4Runner", FailingRunner)
    args = SimpleNamespace(
        checkpoint="checkpoint",
        weight_index=None,
        expert_cache_dir=None,
        max_seq_len=16,
        max_layers=1,
        backend="worker",
        platform="a2a3",
        device=0,
        profile=False,
        verbose_layer_log=False,
    )

    with pytest.raises(RuntimeError, match="runner init failed"):
        generate._create_runner(args)

    assert backend.closed is True
