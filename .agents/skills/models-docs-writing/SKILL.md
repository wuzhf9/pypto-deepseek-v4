---
name: models-docs-writing
description: Create, update, or review code-backed user documentation under docs/models for this DeepSeek V4 Flash PyPTO repository. Use when documenting model architecture, kernels, tensor interfaces, official-to-PyPTO mappings, implementation differences, golden references, precision criteria, acceptance methods, or integration coverage.
---

# Models Docs Writing

## Goal

Write stable user documentation that explains the current model implementation from code evidence. Keep `docs/models/README.md` as the overview and use numbered files such as `01_rmsnorm.md` for module documents.

## Load the writing contract

Before editing a module document:

1. Read [references/document-contract.md](references/document-contract.md) completely.
2. Classify the target with [references/module-types.md](references/module-types.md) and read the matching profile.
3. For a new document, start from [assets/module-document-template.md](assets/module-document-template.md), then remove all comments and non-applicable placeholders.
4. Treat `docs/models/01_rmsnorm.md` as the first concrete style example, not as a source of facts for other modules.

Do not apply the module template to `docs/models/README.md`; preserve its overview and navigation role.

## Follow the evidence workflow

### 1. Establish scope

- Identify the official class, function, or expression being documented.
- Identify the implementation files, callers, tests, and serving integration.
- Decide whether the document describes a primitive, component, execution path, or composite module.

### 2. Inspect current code

Read the relevant sources directly. Search with `rg` before concluding that a symbol is used or unused.

Use this evidence priority:

1. `official/model.py` for official model semantics.
2. `models/config.py` for current model constants and supported dimensions.
3. `models/<module>.py` for PyPTO implementation and standalone validation.
4. Other `models/` files for actual callers, fusion, and orchestration.
5. `tests/models/` for host-side and integration coverage.
6. `serving/` for whole-model runtime integration.

Use `reference/` only as historical context when explicitly requested. Never use it as proof of current behavior.

### 3. Classify every mapping

Label official-to-current relationships precisely:

- **Direct call**: the current path calls the documented kernel.
- **Fused inline**: equivalent math is embedded in a larger kernel.
- **Semantic equivalent**: the math matches but the implementation boundary differs.
- **Available but unused**: code exists but the current model path does not call it.
- **Unsupported/not executed**: the official feature is outside the current runtime path.

Do not describe semantic correspondence as a direct call. Do not infer runtime use from a definition or export alone.

### 4. Document criteria and acceptance methods only

- Derive dtype, shape, `atol`, `rtol`, allowed error ratio, NaN/Inf rules, and commands from code.
- Keep precision criteria and reproducible acceptance commands in module documents.
- Include hardware acceptance commands only; do not include simulator platforms or commands.
- Do not record commit, date, device instance, timing, or observed PASS/FAIL status in module documents.
- If validation is executed for the task, report the result to the user outside the document.
- Keep repository-wide precision caveats in `docs/models/README.md`; do not repeat them in every module document.
- Keep standalone kernel validation, host integration tests, and full-model validation distinct.

Use the repository's `ascend-pypto-validate` skill when the task explicitly requires remote NPU validation.

### 5. Write for users

- Write explanatory prose in Chinese; preserve code identifiers and established technical terms in English.
- Describe current behavior, not the development chronology.
- Link to repository files with relative Markdown links.
- Prefer tables for mappings, interfaces, variants, and acceptance criteria.
- Explain why an implementation differs only when that helps users understand behavior or constraints.
- Avoid line-number references because they become stale quickly.

### 6. Verify the document

- Re-run `rg` for every usage or non-usage claim that materially affects the mapping.
- Confirm every linked local file exists.
- Confirm every command uses current CLI names and arguments.
- Run `git diff --check`.
- Run code tests when code changed or when the user requests validation.
- Report document-only verification separately from runtime validation.

## Preserve scope

Do not change model code merely to make documentation simpler. If code and intended documentation disagree, report the discrepancy and ask whether to document or change the implementation unless the user's request already authorizes the code change.
