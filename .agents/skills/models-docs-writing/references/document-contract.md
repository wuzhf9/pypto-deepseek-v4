# Module document contract

## Purpose

Use this contract for numbered module documents under `docs/models/`. Keep the section sequence stable so readers can move between documents predictably. Specialize heading nouns when useful, but preserve each section's purpose and order.

## Required section sequence

### 1. Module positioning

Use a heading such as `## 模块定位`.

- Explain the module's role in one or two paragraphs.
- State where it appears in the model data flow.
- Include a mathematical definition only when it materially clarifies the behavior.
- Link configuration constants to `models/config.py` rather than duplicating unexplained values.

### 2. Official definition

Use a heading such as `## 官方模型中的 <Module>`.

- Enumerate relevant classes, fields, functions, or inline expressions in `official/model.py`.
- Distinguish learned modules from similar unparameterized math.
- Include official features not executed by the current runtime when omission would otherwise imply feature parity.

### 3. PyPTO implementation

Use `## PyPTO 实现` or a specialized heading such as `## PyPTO kernel 实现`.

- List kernels, wrappers, builders, aliases, and variants that actually exist.
- Explain which symbols are reusable inline functions and which are top-level validation entrypoints.
- Do not treat aliases as separate implementations.

### 4. Official-to-current mapping

Use a heading such as `## 官方模块到当前实现的映射`.

Include a table with at least:

| Official computation | Current implementation | Relationship/status |
|---|---|---|

Use the mapping vocabulary from `SKILL.md`: direct call, fused inline, semantic equivalent, available but unused, or unsupported/not executed.

### 5. Data interface

Use `## 数据接口`.

- Document input, output, state, weight, shape, dtype, and dynamic dimensions.
- State fixed constraints such as batch size, tile divisibility, supported ratios, or sequence assumptions.
- Separate public kernel arguments from internal scratch tensors.
- State checkpoint/runtime dtype conversion only when relevant.

### 6. Implementation method

Use `## 实现方式` or a specialized heading such as `## Kernel 实现方式`.

- Explain the data flow and important computation stages.
- Document tiling, accumulation dtype, rounding, state updates, fusion, or dispatch when user-visible or relevant to validation.
- Avoid translating source code line by line.

### 7. Differences and limitations

Use `## 实现差异与限制`.

- Compare the generic official path with the current fixed-shape or fused implementation.
- State unsupported modes and runtime exclusions explicitly.
- Distinguish an intentional runtime constraint from an unimplemented feature.

### 8. Golden reference

Use `## Golden 参考实现`.

- Identify the golden function and its input snapshot.
- Explain accumulation and output dtype.
- Note ignored outputs, special index comparison, masks, or valid regions.
- If no standalone golden exists, state which higher-level reference is used.

### 9. Precision acceptance

Use `## 精度验收标准`.

- Record `atol`, `rtol`, allowed mismatch ratio, illegal-value rules, and special comparisons.
- Express the effective elementwise condition when it helps readers.
- Derive all values from current code.
- Do not place observed PASS/FAIL results in the document.

### 10. Acceptance method

Use `## 验收方法`.

- Give current hardware commands and compile-only usage as applicable.
- Do not include simulator platforms or simulator commands.
- Include representative boundary shapes when the code specifically supports dynamic or non-tile-aligned inputs.
- Explain relevant flags or prerequisites needed to reproduce validation.
- Do not include observed status, commit, date, device instance, timing, or result tables.
- Keep repository-wide intermittent precision caveats in `docs/models/README.md`, not in module documents.

### 11. Integration coverage

Use `## 集成验证范围`.

- List direct standalone coverage and higher-level tests separately.
- Explain what integration tests cover and what they cannot replace.
- Link to serving integration when the module owns runtime state or lifecycle behavior.

## Optional final section

Add `## 相关文档` only when other user documents already exist and materially help navigation. Do not link to historical `reference/` plans from user documentation by default.

## Style rules

- Use Chinese prose and exact source identifiers.
- Use relative repository links, for example `../../models/rmsnorm.py`.
- Use tables for repeated mappings; use code blocks for interfaces and commands.
- Define abbreviations on first use unless they are already established by `docs/models/README.md`.
- Avoid claims such as “fully aligned,” “all supported,” or “precision passed” without explicit scope and evidence.
- Avoid implementation history, abandoned versions, and planned work unless the document is explicitly a roadmap.
- Keep headings descriptive; do not number headings inside numbered filenames.

## Completion checklist

- All required section purposes are present in order.
- Official and current behavior are clearly separated.
- Direct calls and fused equivalents are not conflated.
- All shapes, dtypes, constants, tolerances, commands, and test names match code.
- Precision criteria and acceptance commands match current code.
- The module document contains no observed validation status or result record.
- All local links resolve.
- `git diff --check` passes.
