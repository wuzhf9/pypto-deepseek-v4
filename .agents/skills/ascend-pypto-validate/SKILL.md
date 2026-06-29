---
name: ascend-pypto-validate
description: Validate PyPTO kernel or model scripts on the remote Ascend NPU server. Use when Codex needs to sync local project files to the Ascend server, run source set_env.sh, execute task-submit with a Python validation script, compare against golden outputs, or rerun with alternate shape arguments such as non-tile-aligned sequence lengths.
---

# Ascend PyPTO Validate

Use this skill to validate PyPTO scripts from this repository on the remote Ascend environment. Keep the workflow generic: the target script, copied files, command arguments, and extra validation cases come from the current task.

## Defaults

- Remote host: `ascend_server`
- Remote project directory: `dsv4`
- Remote environment setup: `source set_env.sh`
- Submit wrapper: `task-submit --device auto --run "<command>"`
- Common run shape: `python <script> -p a2a3 -d {}`
- Local project root: current repository root

Override these defaults if the user or current repository context specifies a different host, directory, environment file, platform, device selector, or command.

## Workflow

1. Identify the validation target.
   - Determine the local script path, for example `models/rmsnorm.py`.
   - Determine whether related local files also changed and must be synced, such as `models/golden.py`, `models/config.py`, or helper modules imported by the script.
   - Determine the base validation command. Prefer the script's own CLI if it exists.

2. Run a local syntax check before using the remote machine.
   - For Python scripts, run `python -m compileall <script-or-package>`.
   - Fix local syntax errors before syncing.

3. Sync the required files to the remote project directory.
   - Use `scp <local-path> ascend_server:dsv4/<same-relative-path>` for one or a few files.
   - Preserve the repository-relative path on the remote side.
   - Sync imported helper files when the remote copy may be stale.

4. Execute the validation on Ascend.
   - Run:

```bash
ssh ascend_server 'source set_env.sh && cd dsv4 && task-submit --device auto --run "python <script> -p a2a3 -d {}"'
```

   - Keep the full command in one SSH invocation so the environment setup and working directory apply to the submitted task.
   - If the command needs braces for a device placeholder, preserve `-d {}` inside the quoted `--run` command.

5. Interpret the result.
   - Treat `[RUN] PASS` and task exit code `0` as success.
   - If compilation fails, fix the PyPTO API usage or shape/type issue reported in the log.
   - If runtime validation fails, inspect the first mismatches, shape arguments, valid-shape handling, dtype casts, and golden comparison tolerance.

6. Run shape-sensitive follow-up cases when relevant.
   - For dynamic sequence or token dimensions, run at least one non-tile-aligned case, for example:

```bash
ssh ascend_server 'source set_env.sh && cd dsv4 && task-submit --device auto --run "python <script> -p a2a3 -d {} -s 13"'
```

   - Choose the extra argument from the target script's CLI. Do not assume `-s` exists for every script.

## Reporting

Report the exact validation commands that matter and the pass/fail outcome. If validation fails, include the first actionable error line and the next fix to try. Do not describe unrelated Ascend environment warnings unless they block execution.
