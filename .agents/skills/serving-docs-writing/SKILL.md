---
name: serving-docs-writing
description: Create, update, or review code-backed user documentation under docs/serving for this DeepSeek V4 Flash PyPTO repository. Use when documenting serving entrypoints, CLI workflows, checkpoint or expert-cache preparation, whole-model orchestration, Host/NPU data flow, runtime values, device residency, allocation lifetimes, profiling, validation methods, or serving constraints.
---

# Serving Docs Writing

## Goal

Write stable user documentation that explains the current serving implementation from code evidence. Keep `docs/serving/README.md` as the overview and use numbered files such as `01_generate.md` for workflow and runtime-component documents.

## Load the writing rules

Before editing a serving document:

1. Read [references/document-contract.md](references/document-contract.md) completely.
2. Classify the target with [references/document-types.md](references/document-types.md) and read the matching profile completely.
3. For a new workflow document, start from [assets/workflow-document-template.md](assets/workflow-document-template.md).
4. For a new runtime-component document, start from [assets/runtime-component-template.md](assets/runtime-component-template.md).
5. Remove all template comments and non-applicable placeholders.

Do not apply either topic template to `docs/serving/README.md`. Follow the overview profile and checklist in `references/document-types.md` instead.

## Follow the evidence workflow

### 1. Establish scope

- Identify whether the target is the overview, a user workflow, or a runtime component.
- Identify its implementation files, callers, downstream dependencies, tests, and CLI integration.
- Define what the document owns and what belongs in a linked document.

### 2. Inspect current code

Read relevant sources directly. Search with `rg` before concluding that a path, option, symbol, cache, or cleanup behavior exists or is unused.

Use this evidence priority:

1. Root entrypoints such as `generate.py`, `smoke_model.py`, and `export_expert_cache.py` for current CLI behavior.
2. `serving/<module>.py` for the target implementation.
3. Other `serving/` files for actual callers, ownership, lifetime, and integration.
4. `models/` public spec builders and kernel entrypoints only when confirming a serving binding.
5. `tests/cli/` and `tests/serving/` for current validation coverage and error cases.

Use `reference/` only as historical context when explicitly requested. Never use it as proof of current behavior.

### 3. Trace data and ownership

For every important value, state its actual representation and location:

- Host `torch.Tensor`;
- runtime-owned device tensor;
- fixed `RuntimeWeight`;
- transient `HostStagingTensor`;
- persistent state;
- reusable intermediate or scratch allocation;
- on-disk expert cache data.

Distinguish creation, H2D/D2H transfer, reuse, commit, release, and close. Do not call every reuse mechanism a cache.

### 4. Separate prefill and decode

- Describe prefill and decode separately whenever their kernels, routed-expert loading, state access, or data transfers differ.
- Do not generalize prefill routed packs and decode selected-expert staging into one weight path.
- Explain the decode pre-MoE/post-MoE split only from the current runner and split-block bindings.

### 5. Document validation methods only

- Derive commands, required files, arguments, shapes, dtypes, state rules, and expected invariants from code and tests.
- Keep unit tests, Host integration tests, smoke execution, and full hardware execution distinct.
- Include current hardware commands only; do not include simulator commands.
- Do not record commit, date, device instance, timing result, or observed PASS/FAIL status.
- If validation is executed for the task, report its result to the user outside the document.

Use the repository's `ascend-pypto-validate` skill when the task explicitly requires remote NPU validation.

### 6. Write for users

- Write explanatory prose in Chinese; preserve code identifiers and established technical terms in English.
- Describe current behavior, not development history or abandoned backends.
- Link repository files with relative Markdown links and avoid line-number references.
- Prefer tables for CLI arguments, public interfaces, allocation categories, cache layers, and validation methods.
- Link to another serving document instead of duplicating its detailed explanation.
- Introduce terms in `docs/serving/README.md` before relying on them across multiple documents.

### 7. Verify the document

- Re-run `rg` for every material usage, fallback, residency, or cleanup claim.
- Confirm every linked local file exists.
- Confirm every CLI command uses current names and arguments.
- Confirm cache terminology and Host/NPU placement match the implementation.
- Run `git diff --check`.
- Run code tests when code changed or when the user requests validation.
- Report document-only verification separately from runtime validation.

## Preserve scope

Do not change serving or model code merely to simplify documentation. If the implementation contradicts the intended documentation, report the discrepancy and ask whether to document or change it unless the user's request already authorizes code changes.
