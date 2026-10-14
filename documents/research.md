# Research direction

## Proposal

- Investigate flow matching training with a one-to-one mapping between each condition and its target sample.
- The user hypothesizes that sufficiently informative conditions enable time-constant trajectory velocities and native one-step generation; this remains a research hypothesis.
- The proposed trade-off is a more informative or denser condition than conventional conditioning; the first toy experiment is specified in the [HTML development plan](experiment-01-plan.html).

## First experiment

- The user requests a 2D Gaussian prior, a spiral target, a conditional MLP, continuous-time training, and Euler inference with one-step performance as the primary focus.
- Use 50,000 fixed target points; the proposed control is the number of ordered groups K, ranging from K=1 to one condition per target while model size and target data stay fixed.
- Represent each integer group index j as the normalized FP32 value `c = 2(j + 0.5)/K - 1`; this is one scalar MLP input with a consistent geometric order across densities.
- Step 1 implements the spiral, condition encoding, independent pair sampling, and reference visualization in `empirical_condition_density/`; it is awaiting level-1 review.
- Data will be built in memory during the experiment; the unified YAML configuration and Bash launcher are deferred until the full experiment code is assembled.
- Model training and evaluation have not started; the remaining training and evaluation settings are proposed defaults.

## Mathematical interpretation

- For analysis, assume a condition determines its target as `x_1 = f(c)` and use the linear path `x_t = (1-t)x_0 + t f(c)`; the path choice is an analytical assumption, not a finalized design.
- For `t < 1`, the ideal condition-dependent field is `v*(x,t,c) = (f(c)-x)/(1-t)`; evaluated along the corresponding path, it equals the constant displacement `f(c)-x_0`.
- A perfect model therefore reaches `f(c)` in one Euler step from `t=0`; the field still depends on position and time, and velocities differ across prior samples sharing the same condition.

## Questions to resolve

- Unique condition-target pairs in a finite dataset do not establish a deterministic relationship for unseen conditions; condition dimensionality alone does not establish sufficient information.
- Determine how conditions are obtained at inference and account for their generation cost; a condition extracted from the target requires an inference-time source.
- If the target is fully determined by the condition, varying the prior sample cannot change the ideal output; diversity must come from varying the condition.
- Compare against direct prediction of `f(c)` to establish the benefit of flow matching training; finite-model one-step accuracy remains to be evaluated.

## Reference

- [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) provides the conditional/marginal field framework; the deterministic-condition analysis above is our derivation under the stated assumptions.
