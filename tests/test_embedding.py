"""Tests for embedding golden logic against official ``ParallelEmbedding``."""

import importlib

import pytest
import torch

import models.embedding as embedding


official_model = importlib.import_module("official.model")

VOCAB = 32
HIDDEN = 16
SEQ_LENS = [1, 5, 13]


@pytest.fixture()
def tiny_embedding(monkeypatch):
    monkeypatch.setattr(official_model, "world_size", 1)
    monkeypatch.setattr(official_model, "rank", 0)
    monkeypatch.setattr(embedding, "VOCAB", VOCAB)
    monkeypatch.setattr(embedding, "HIDDEN", HIDDEN)

    with official_model.set_dtype(torch.bfloat16):
        module = official_model.ParallelEmbedding(VOCAB, HIDDEN)

    with torch.no_grad():
        weight = torch.arange(VOCAB * HIDDEN, dtype=torch.float32).reshape(VOCAB, HIDDEN)
        module.weight.copy_((weight * 0.01 - 1.0).to(torch.bfloat16))
    return module


def _input_ids(seq_len: int) -> torch.Tensor:
    values = torch.tensor([0, VOCAB - 1, 7, 7, VOCAB // 2, 3, VOCAB - 2], dtype=torch.int64)
    repeats = (seq_len + values.numel() - 1) // values.numel()
    return values.repeat(repeats)[:seq_len].view(1, seq_len)


@pytest.mark.parametrize("seq_len", SEQ_LENS)
def test_golden_embedding_matches_official_parallel_embedding(tiny_embedding, seq_len: int) -> None:
    input_ids = _input_ids(seq_len)

    with torch.no_grad():
        h = tiny_embedding(input_ids)
        expected = h.unsqueeze(2).repeat(1, 1, embedding.HC_MULT, 1)

    tensors = {
        "input_ids": input_ids.clone(),
        "weight": tiny_embedding.weight.detach().clone(),
        "out": torch.zeros(1, seq_len, embedding.HC_MULT, HIDDEN, dtype=torch.bfloat16),
    }
    embedding.golden_embedding(tensors)

    torch.testing.assert_close(tensors["out"], expected, rtol=0, atol=0)


def test_build_embedding_specs_shapes_and_dtypes(monkeypatch) -> None:
    monkeypatch.setattr(embedding, "VOCAB", VOCAB)
    monkeypatch.setattr(embedding, "HIDDEN", HIDDEN)
    seq_len = 5

    specs = embedding.build_embedding_specs(seq_len)
    tensors = {spec.name: spec.create_tensor() for spec in specs}

    assert tensors["input_ids"].shape == (1, seq_len)
    assert tensors["input_ids"].dtype == torch.int64
    assert tensors["weight"].shape == (VOCAB, HIDDEN)
    assert tensors["weight"].dtype == torch.bfloat16
    assert tensors["out"].shape == (1, seq_len, embedding.HC_MULT, HIDDEN)
    assert tensors["out"].dtype == torch.bfloat16
