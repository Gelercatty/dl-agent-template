---
name: dl-experiment
description: Prepare, validate, run and document reproducible deep-learning experiments in this repository using frozen Releases, generated Jobs and separate Runs. Use for training workflow changes, experiment preparation, resume and result review; not for unrelated research or cloud administration.
---

# DL experiment workflow

Read the repository [AGENTS.md](../../../AGENTS.md) and
[workflow guide](../../../docs/workflow.md). Resolve commands from the repository root, not this skill
directory. This skill is repository-local and refers to that repository's maintained documentation.

Locate the designated development checkout and inspect its changes. Local or remote development is
valid. Preserve unknown edits and identify which tree is the source before transferring code.

For an experiment, turn the user's question into a complete registered config and a concrete plan.
Use `python -m workflow release`, then `prepare`, then `check`; pass the generated Job path rather than
launching from mutable source. Inspect the configuration, source identity, resource needs, resume
source and output location. Use an appropriate smoke check for a new trainer.

Preparation and dry-run are separate from training or platform submission. Apply authorization
already given for this task. If execution authorization is missing, prepare the reviewable job first,
then ask only for the missing action; do not request the same authorization again.

For trainer changes, read the [adapter contract](../../../docs/adapters.md), preserve the existing
framework and implement real output validation. Do not claim distributed/GPU readiness from CPU
checks. Bind infrastructure outside tracked source.

For recovery, inspect the exact Job and Run, terminal markers and complete checkpoint generations.
No terminal marker means unfinished/unknown, not success. Resume through a new Job, retaining the old
run and verifying checkpoint/config compatibility; never patch a frozen definition.

For result review, use `python -m workflow status --job ...`, read metrics and selected evidence, then
compare with the original plan. Follow the [records contract](../../../docs/records.md): formal
training requires pre/post records; ordinary investigation stays in chat unless recording is requested.
Keep the original hypothesis, record deviations and separate execution success from task quality.
