# CPU checkpoint continuation

- Question: does checkpoint recovery preserve the full optimizer and random state?
- Baseline: the `demo` pipeline at step 40.
- Change: run `demo-long` to total step 80, resuming from the baseline checkpoint.
- Keep model, data generation, seed, optimizer and all other config fields identical.
- Acceptance: 80 completed steps, resumed_from_step = 40, valid checkpoint and improved loss;
  the integration test additionally compares every model tensor against an uninterrupted 80-step run.
- Resources: one CPU process; no GPU or downloaded dataset.
- Failure action: retain both runs; use a new Job ID for another attempt.
