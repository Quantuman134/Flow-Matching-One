# Research direction

## Hypothesis

- A condition that uniquely determines its target may enable native one-step flow matching; the trade-off is more informative conditioning.
- This is a research hypothesis; finite-model accuracy and benefits over direct prediction remain unverified.

## Linear-path interpretation

- Let `x0` be prior noise at `t=0`, `c` a condition, and `f(c)` its unique target at `t=1`; assume the path `x_t = (1-t)x0 + t f(c)`.
- For `t<1`, the ideal velocity field is `v*(x,t,c) = (f(c)-x)/(1-t)`; along a sampled path it equals the constant displacement `f(c)-x0`.
- One exact Euler update at `t=0` reaches `f(c)`; the field still depends on position and time, and different noise samples have different velocities.

## Open questions

- Generalization: unique training pairs do not establish a deterministic mapping for unseen conditions.
- Condition source: how are informative conditions obtained at inference, and what do they cost?
- Diversity: if a condition fixes its target, diversity must come from varying conditions.
- Baseline: compare with direct prediction of `f(c)` to isolate the benefit of flow matching training.

## Related work

- Framework: [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747); the deterministic-condition interpretation above is our derivation.
- First study: [condition-density experiment plan](experiment-01-plan.html); implementation and review status live in the [topic README](../empirical_condition_density/README.md).

## Structured source question

- Can a clean-image source supply enough content information for accurate one-step brightness transport, with brightness as the explicit condition?
- This differs from a target uniquely determined by its condition: the target depends on both source image and brightness. Benefits over direct residual regression and sensitivity to training-time coverage remain open.
- Design and proposed settings: [brightness experiment plan](experiment-02-brightness/plan.html).
