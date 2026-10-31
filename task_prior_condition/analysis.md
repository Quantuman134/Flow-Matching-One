# Task-related sources and informative conditions

Date: **2026-10-28**. Status: literature analysis and proposed research plan; no new training results.

Literature update **2026-10-30**: [benchmark candidates](benchmark_candidates.md) adds task-specific access checks and close comparisons including LBM and Marigold V2.

## Assessment

Both directions are worth investigating together. A task-related source can preserve content, and a condition can specify the transformation or resolve the remaining ambiguity. The stronger research question is: **which source–condition representation gives accurate one-step transport at the lowest total inference cost, and when does flow-matching training improve on direct residual prediction?**

The broad ingredients already have substantial precedent. A contribution will need a useful representation design, evidence explaining when it works, and an advantage in accuracy, generalization, control, or total cost. One paired training example per input establishes an empirical pairing; determinism must also hold for the intended inference setting.

## Current evidence

The completed spiral baseline used 50,000 targets, seven separately trained models, 20,000 updates each, and one seed. Conditions encode ordered position along the spiral. Evaluation uses training conditions with fresh source noise; this assesses noise generalization and fit at those conditions.

Geometric precision is the fraction of outputs within distance 0.05 of any target; conditional precision restricts this check to targets belonging to the requested condition. Both range from 0 to 1, with larger values indicating more valid outputs.

| Conditions | Targets per condition | One-step geometric precision | One-step conditional precision |
| --- | --- | --- | --- |
| 1 | 50,000 | 0.00680 | 0.00680 |
| 100 | 500 | 0.73276 | 0.70720 |
| 500 | 100 | 0.73038 | 0.57064 |
| 50,000 | 1 | 0.73152 | 0.39654 |

The reference RMSE is the square root of the mean squared Euclidean distance from a one-step output to its condition's target-group mean, in normalized spiral-coordinate units; smaller is better. It is 0.08361 at 50,000 conditions, where the reference is the exact requested target, versus 0.08276 at 100 conditions.

At 50,000 conditions, four-step conditional precision rises to 0.89574. Thus, more informative conditions improve the broad one-step distribution initially, but the trained models leave substantial approximation or optimization error; condition uniqueness does not establish accurate one-step inference. Conditional precision also becomes stricter as groups shrink, so its values alone do not measure how optimization difficulty changes across condition counts.

Evidence: [summary table](../empirical_condition_density/outputs/comparison/summary.csv), [metric definitions](../documents/experiment-01-plan.html), and [run status](../empirical_condition_density/README.md). These single-seed results support a mechanism study, with uncertainty across training runs still to measure.

The brightness experiment already addresses the first question: a clean image is the structured source, and exposure specifies its target. Its full training-timestep/NFE comparison remains pending; short learning checks and resource pilots establish implementation feasibility. The existing one-timestep setting always trains at zero and is already the direct residual-regression baseline. See the [brightness plan](../documents/experiment-02-brightness/plan.html) and [status](../imagenet_brightness_flow/README.md).

## Mathematical setting

The following endpoint analysis is our derivation for paired, linear-path, squared-error FM. It uses the standard vector-field regression framework of [Flow Matching](https://arxiv.org/abs/2210.02747).

| Symbol | Meaning |
| --- | --- |
| S | Task-related starting sample in the transport space |
| C | Condition available during inference |
| Y | Paired target in the same space as S |
| t | Transport time, zero at source and one at target |
| X_t | Interpolated training state |
| v_theta | Predicted instantaneous velocity |

Train on the linear path and predict its displacement:

\[
X_t=(1-t)S+tY,\qquad
\mathcal L_{\rm FM}=\mathbb E\|v_\theta(X_t,t,C)-(Y-S)\|_2^2.
\]

At the source boundary, the squared-error optimal velocity is

\[
v^*(s,0,c)=\mathbb E[Y\mid S=s,C=c]-s,
\qquad
\widehat Y_{\rm one}=s+v^*(s,0,c)=\mathbb E[Y\mid s,c].
\]

This is the boundary regression optimum; continuous-time training relies on learning its limiting behavior near zero. Explicit supervision at zero can make that requirement easier to optimize.

The needed property is **joint target determination**, \(Y=T(S,C)\), or sufficiently small residual uncertainty for the requested accuracy. Global invertibility is unnecessary: the current exposure operator clips highlights, which permits several source values to produce the same target value while retaining a deterministic forward map.

The paired endpoint MSE separates information loss from prediction error. Here the first term measures remaining target ambiguity and the second measures fit to the conditional mean; both use summed squared transport-coordinate units, with smaller values better:

\[
\mathbb E\|Y-\widehat Y_\theta(S,C)\|^2
=\mathbb E\,\mathrm{tr}\,\mathrm{Cov}(Y\mid S,C)
+\mathbb E\|\mathbb E[Y\mid S,C]-\widehat Y_\theta(S,C)\|^2.
\]

This identity concerns paired squared error. One-step generative methods can represent multimodal distributions by learning other objectives or couplings; [MeanFlow](https://arxiv.org/abs/2505.13447) learns average velocity over a time interval and provides an essential comparison.

### Path information

Unique endpoints do not guarantee that the entire linear interpolation admits a single-valued velocity field given only \((X_t,t,C)\). For example, let equally likely sources be \(S=-1,+1\), let \(Y=-S\), and hold C fixed. The endpoint map is bijective, but both paths pass through zero at \(t=1/2\), with velocities +2 and -2.

Supplying the unchanged source as additional context resolves this conflict: a deterministic target allows \(v^*(X_t,t,S,C)=T(S,C)-S\) along each path. Compare this architecture with one that receives the source only through the changing state. Endpoint sufficiency and velocity ambiguity along the path are separate quantities.

The current SiT receives changing state, time, and scalar exposure. Its encoded source also passes through a lossy VAE: deterministic image-space exposure does not by itself establish exact target determination from the source latent. Target reconstruction through the VAE is a useful representation reference, although it is not a mathematical lower bound on every possible decoded prediction.

### Direct prediction

At time zero, the residual objective

\[
\mathcal L_0=\mathbb E\|v_\theta(S,0,C)-(Y-S)\|^2
\]

is exactly direct residual regression. Full-path FM may offer useful training augmentation, initialization, regularization, or iterative refinement, but its empirical advantage must be measured. The research value could instead lie in the representation and its control or transfer properties if direct regression performs equally well.

## Related work

| Work | Relevant contribution | Implication for this project |
| --- | --- | --- |
| [Rectified Flow](https://arxiv.org/abs/2209.03003) | Transport between arbitrary observed distributions and trajectory straightening. | Task-related endpoints and efficient transport have established foundations. |
| [LBM](https://github.com/gojasper/LBM), ICCV 2025 | Single-step latent bridge image translation, including relighting and dense geometry prediction, with public code. | A required close baseline for a general one-step prediction claim. |
| [pix2pix-Turbo](https://arxiv.org/abs/2403.12036), 2024; [official code](https://github.com/GaParmar/img2img-turbo) | One-step image translation. | Consult its designs and training techniques when designing our method, as suggested by the user. |
| [BBDM](https://openaccess.thecvf.com/content/CVPR2023/papers/Li_BBDM_Image-to-Image_Translation_With_Brownian_Bridge_Diffusion_Models_CVPR_2023_paper.pdf) and [I2SB](https://i2sb.github.io/) | Learn bridges between image domains or degraded and clean images. | Starting from an informative image is an established approach. |
| [RenderFlow](https://arxiv.org/html/2601.06928v1) | Uses albedo as source, geometry/material buffers and an environment map as conditions, stochastic bridge training, pretrained Wan2.1, and optional reference keyframes. | Strong application precedent; isolate source design from pretraining, auxiliary losses, and additional guidance. |
| [Better Source, Better Flow](https://arxiv.org/html/2602.05951v1), February 2026 | Learns a condition-dependent source, analyzes intrinsic FM variance, and addresses source collapse and instability. | A close comparator for learned-source design; justify the task representation and one-step objective beyond this framework. |
| [SubFlow](https://arxiv.org/html/2604.12273v1), April 2026 | Adds semantic sub-mode conditions to existing one-step frameworks, including MeanFlow. | Finer conditioning to reduce averaging is also established; unimodality still permits within-mode variation. |
| [One-Step FM for Path-Dependent Physical Fields](https://arxiv.org/html/2606.22752v1), June 2026 | Pairs a geometry-based source with stress fields and embeds loading histories as conditions. | A close example of abstract task sources with explicit transformation information. |
| [PhiZero](https://arxiv.org/html/2607.28624v1), July 2026 | Learns discrete transition tokens from videos, predicts them with a reasoner, and renders with first-frame appearance context and a standard FM diffusion decoder. | Useful representation precedent; measure token sufficiency, prediction errors, and total generation cost. |
| [MeanFlow](https://arxiv.org/abs/2505.13447) | Learns average velocity for native one-step sampling. | Controls whether a dedicated one-step objective explains the gain. |

RenderFlow's input contains more than albedo and lighting; its screen-space information can still omit scene content needed for exact light transport. Its empirical one-step quality therefore supports feasibility without proving exact one-to-one physical reconstruction.

PhiZero's physical language is a learned quantized representation of transitions. Its training extracts tokens using the observed video; deployment predicts them from available inputs. Our inference is that this factorization may help one-step rendering, but sufficient target information and accurate one-step decoding require separate experiments.

For accessible intuition, the [MeanFlow author's CMU blog](https://www.cs.cmu.edu/~csd-phd-blog/2026/mean-flows/) explains why average and instantaneous velocity lead to different sampling behavior. Technical claims here rely on the linked papers and project sources. This search identifies close precedents; novelty remains a claim to establish for the final method.

## Question 1: task-related source

**Worth pursuing:** tasks where content or state already exists, transformation controls are available, and deterministic, fast output matters. Rendering, controlled edits, and fully specified physical-state prediction are plausible settings, subject to task-specific information checks.

For content-preserving paired tasks, the source must correspond to the requested target. Merely sampling an independent image from a visually related marginal distribution supplies little information about that target. A learned or procedural source must also be cheap enough to justify its generation cost.

The research opportunity is to establish a reproducible rule for selecting sources and conditions, or a representation that improves transfer or accuracy under a fixed budget. A demonstration that informative paired inputs permit a one-pass predictor is a useful baseline result with limited novelty by itself.

## Question 2: abstract source and informative condition

**This is a promising extension of Question 1.** Let I denote an available input image or state and u the requested operation. Construct an abstract source \(S=A(I)\) and a condition \(C=H(I,u)\), then predict \(Y=T(I,u)\). Appearance, geometry, motion, and transformation parameters can be routed separately.

| Source abstraction | What the condition must supply when needed |
| --- | --- |
| Grayscale or low-frequency appearance | Relevant color and texture information plus the edit controls |
| Geometry or object-state layout | Appearance/material references and specified transformation or dynamics |
| Coarse warped frame | Motion specification, occlusion information, and appearance references |

A useful condition has clear task content: object-linked trajectories, spatial masks, numerical parameters, ordered operations, or learned transition tokens. Token count alone measures capacity poorly; informativeness, alignment, and model access matter.

Abstraction discards information. If C receives the discarded information exactly, the experiment studies routing and inductive bias. If C compresses it, the experiment studies an accuracy–capacity trade-off. Gains can come from easier computation, reuse across outputs, or better generalization, even when the total information stays constant.

If multiple futures remain valid, a planner can sample a transition code and the renderer can produce the corresponding outcome. This moves some uncertainty into plan selection. Evaluate diversity and coverage across plans, as well as uncertainty remaining within each plan.

## Proposed study

Working question: **Can task structure be divided between source and condition so that one renderer evaluation achieves useful accuracy and generalization at lower total cost?** The stages below are proposals; they preserve the existing experiment review process.

### 1. Complete the current diagnostic

Finish the approved brightness workflow and evaluate the existing continuous/16/4/1 training-time settings at NFE 1/4/16. The always-zero model already tests residual regression; mark evaluations at times outside its training support.

Use the analytic exposure operator and target VAE reconstruction as reference outputs. Brightness has a cheap exact solution and therefore serves as a diagnostic of training and representation rather than an application-speedup claim. Inspect held-out images, zero exposure, saturation strata, and source/target latent errors.

Interpretation: a continuous-time advantage motivates further study of path training. A tie puts the evidence behind representation design. Poor results across settings motivate checking conditioning, training convergence, and the representation before scaling.

### 2. Separate ambiguity from learning

Extend the small controlled setting with three factors: source information, condition information, and availability of unchanged-source context. Construct paired examples with known conditional means and variances, including the bijective crossing example above and an ambiguous example where the same available inputs have two valid targets.

Compare direct residual regression, standard full-path FM, and FM with explicit time-zero supervision. Use held-out source values and condition intervals, and repeat shortlisted comparisons over at least three seeds. This establishes whether errors arise from missing information, conflicting path labels, or finite-model fit.

### 3. Test routing and compression on controlled images

A practical first proposal reuses existing images and constructs localized, ordered appearance edits with known masks and numerical parameters. Use nonlinear color/exposure operations and their compositions so operation order matters. Keep every transformed version of a source image in the same data split.

For a strict routing control, encode each original image once into common source features Z. Derive all starting states and reference tokens from Z, using the same context tokenizer. Pooling or projection followed by a fixed lift restores the abstract state to the target shape. Compare these configurations:

| Configuration | Starting state | Persistent condition |
| --- | --- | --- |
| Task source | Original source latent Z | Edit program |
| Abstract source | Pooled/projected source latent | Same edit program and original-source latent tokens |
| Noise source | Gaussian latent | Same edit program and original-source latent tokens |

The source reference is an information-routing control. Next replace it with a source-derived bottleneck at several token budgets. Include a source-plus-reference variant to expose the benefit of persistent context. Image-space grayscale or low-pass sources can follow as separate representation experiments. Match parameter count and training cost where feasible, and report attention/encoding overhead when token counts differ.

Compare structured controls against a coarser description and a same-information alternative encoding. Hold out operation compositions and parameter intervals; compare condition swaps on a fixed source to test whether the model follows the specified edit. Report both numerical adherence and image preservation.

These synthetic edits are a controlled representation study. An application result should follow on a task with meaningful synthesis or simulation cost and inputs available at deployment.

### 4. Compare objectives and predicted conditions

Screen the representation variants with the direct residual baseline and standard FM. On the strongest variants, add MeanFlow and the endpoint-supervised FM variant. Reuse the same backbone, pretraining status, examples, and comparable compute; report both equal-data and equal-training-time comparisons when objective costs differ.

Begin with conditions supplied directly by the task. If learning transition tokens becomes justified, first assess reconstruction using target-extracted tokens as an oracle, then freeze the token interface and train a predictor from inference inputs. Compare oracle tokens, predicted tokens, and corrupted tokens at the same budget. Training-only target information belongs solely in the oracle assessment.

For video, a later candidate is first-frame appearance plus object trajectories and events, with a cheap warped source. The critical first experiment compares supplied trajectories against predicted trajectories; a large gap indicates that planning is the limiting component. Choose the dataset and backbone after the controlled stages establish a reason to scale.

### 5. Measure value and decide

| Measure | Purpose, calculation, and interpretation |
| --- | --- |
| Endpoint RMSE | Square root of mean squared prediction error; report normalized RGB or latent units separately, with lower values indicating better paired accuracy. |
| Condition adherence | Difference between requested and recovered parameters, mask effects, or trajectories; specify native units such as exposure stops or pixels, with lower errors better. |
| One-step quality gap | Paired error at one evaluation minus error at a chosen multi-step reference for the same model; report original error units, with values near zero indicating little refinement benefit. |
| Conditional uncertainty | Exact expected conditional squared deviation in controlled data; report squared target-coordinate units, with zero indicating target determination. Real-data estimates require an explicit estimator and remain proxies. |
| Path ambiguity | Conditional variance of target displacement given the actual state, time, and context; calculate exactly in controlled examples, in squared transport units, with lower values indicating fewer conflicting labels. |
| Condition capacity | Number of tokens, vocabulary size or scalar precision, and actual encoded size when measured; smaller capacity is useful at matched quality. Token count and nominal bits are capacity descriptors rather than estimates of target information. |
| Total latency | Measured milliseconds per output for preprocessing, planning, source construction, context encoding, rendering, and decoding; lower is better at matched quality. Report single-output and amortized reuse separately. |
| Generalization | Endpoint/adherence errors on held-out sources, compositions, and parameter intervals; report their native scales and the change from the ordinary test split. |

For real synthesis tasks, add perceptual similarity with its exact implementation and a task-relevant structural measure. For multimodal tasks, assess sample quality and coverage across multiple plans; a single paired RMSE cannot establish distribution quality.

Advance when an abstract source and obtainable condition match useful quality at lower total cost, improve held-out composition accuracy, or provide a valuable reusable control interface. Attribute a flow-specific gain only when it survives the matched direct-regression comparison and training-budget controls.

If the direct baseline ties, retain any demonstrated representation benefit and revise the method claim accordingly. If gains depend on target-extracted conditions or expensive planning, establish their practical cost before scaling. If abstraction removes information that conditions cannot recover at the desired budget, use a richer source or explicitly model residual uncertainty.

The immediate recommendation is to finish brightness and develop the small information/path diagnostic in sequence, while treating abstract-source representation design as the longer-term contribution to investigate.
