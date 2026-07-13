# Module type profiles

Select the closest profile before writing. The document contract remains mandatory; these profiles add module-specific content.

## Primitive kernel

Examples: RMSNorm, Linear, RoPE.

Add:

- Mathematical definition or exact transform.
- Supported dimension/shape variants.
- Tiling and reduction strategy.
- Accumulation, rounding, and output dtype.
- Inline kernel versus top-level validation wrapper.
- Direct callers and fused equivalents.

Avoid presenting a fixed-shape kernel family as a generic arbitrary-shape API.

## Model component

Examples: Embedding, Hyper-Connection, attention QKV/output projection, Gate, Expert, Head.

Add:

- Position in the model data flow.
- Weight groups and tensor transformations.
- Intermediate tensors that cross component boundaries.
- Prefill/decode differences when present.
- Any embedded primitive operations and whether they are fused.

Avoid duplicating a primitive document; link to it when a stable user document exists.

## Stateful execution path

Examples: SWA, CSA, HCA, Compressor, Indexer, MoE selected-expert decode.

Add:

- Prefill and decode paths separately.
- Persistent state/cache schema, update timing, and ownership.
- Layer/config conditions selecting the path.
- Dynamic lengths, offsets, masks, and Top-K semantics.
- State initialization, reuse, and boundary behavior.
- Differences between standalone validation state and serving-owned device state.

Avoid describing host plans or cache ownership as kernel-local behavior.

## Composite model layer

Examples: Block and Split Block.

Add:

- Ordered submodule sequence.
- Dispatch matrix by layer type, compression ratio, routing type, and prefill/decode mode.
- Inputs and outputs passed between fused sections.
- State mutations and expert staging boundaries.
- Relationship between full and split implementations.
- Integration-level precision scope.

Avoid repeating every child kernel's internal tiling; link to child documents instead.

## Overview or navigation document

Example: `docs/models/README.md`.

Do not apply the module document section contract. Cover implementation scope, official alignment boundary, directory map, main execution flow, and links to module documents. Keep it shorter than the combined module documentation.
