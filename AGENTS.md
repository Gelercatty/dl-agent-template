# Agent collaboration rules

Read [README.md](README.md) and [docs/workflow.md](docs/workflow.md) before changing this workflow.
For trainer integration, also read [docs/adapters.md](docs/adapters.md); for experiment records,
read [docs/records.md](docs/records.md). The repository skill is
[dl-experiment](.agents/skills/dl-experiment/SKILL.md).

- Keep one designated development checkout per branch. Local and remote development are both valid;
  inspect both sides before synchronizing and preserve unknown/uncommitted changes.
- Keep model code in source, experiment semantics in registered configs, and deployment bindings in
  ignored `.local/` or the environment. Do not commit credentials, private host paths, datasets,
  checkpoints, media, generated job definitions or run logs.
- Formal training uses a verified frozen Release and a generated Job. A release may capture a dirty
  working tree. Fix source/config in Code, then create another Release/Job; do not patch frozen files.
- Temporary processing may run in ignored `scratch/`. Promote reusable scripts and required tests to
  source. Do not create an alternate untracked formal training entrypoint.
- Prepare the complete job and pass dry-run before requesting any missing execution authorization.
  Follow authorization already given for the specific task; do not ask again. Preparing a job alone
  does not authorize GPU allocation, external submission, publishing, or starting a service.
- For ordinary investigation, review and tests, communicate in chat. Write progress/conclusion files
  only when asked to record them. Formal training requires a pre-run plan and a post-run result,
  including failed/interrupted outcomes, without an additional request to document it.
- Preserve the original plan. Append actual deviations, evidence, interpretation and unresolved
  checks to the training record. Distinguish controller success from scientific quality acceptance.
- Run only one controller per Job. Adapters own distributed launch and ensure global-rank-zero,
  atomic output publication. Never infer success from elapsed time or missing error messages.
- Preserve failed runs and checkpoints unless cleanup is explicitly authorized. Old source belongs
  in Git history; do not create a `History/` source-copy pile.
- Before finishing code changes, run `uv run --locked ruff check .`,
  `uv run --locked ruff format --check .` and `uv run --locked --extra demo pytest -q`.
  Report missing dependencies as unverified, not as passing checks. When changing a trainer, add the
  smallest meaningful behavioral check for the changed contract.
- Keep existing trainer/framework choices unless the task calls for replacing them. This template is
  a workflow starter, not a reason to build a new scheduler or generic training framework.
