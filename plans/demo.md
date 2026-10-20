# CPU workflow smoke test

- Question: can this environment execute the frozen code and produce a restorable checkpoint?
- Hypothesis: a small MLP will learn the two seeded synthetic classes within 40 steps.
- Pipeline: `demo`; seed 7; 128 training samples and a separate 128-sample validation split.
- Acceptance: 40 completed steps, finite model and metrics, training loss below its initial value,
  validation accuracy at least 90%, checkpoint/config/step identities matching.
- Resources: one CPU process, one torch thread; no GPU, external data or tracking service.
- Interpretation limit: this only validates the example and workflow, not a production model.
- Failure action: preserve log, terminal marker and any complete checkpoint pair; inspect before retry.
