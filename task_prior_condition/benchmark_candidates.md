# Benchmark candidates

Date: **2026-10-30**. Status: recommendations and access review; no dataset downloads, baseline reproductions, or new experiments. Access statements refer to the official pages inspected, rather than a verified archive download. Proposed sources and conditions are our designs.

This is the initial six-task review. The current pool for user selection is the [60-task catalog](candidate_catalog.md).

## Recommended scope

For a vision paper, develop **deblurring, controlled relighting, and frame interpolation** as the main tasks, with **depth** as an optional dense-prediction test. For a broader prediction paper, consider replacing depth with a **physical forward problem**. Keep brightness and the spiral as mechanism studies.

Suggested execution order: a small GoPro deblurring pilot, MIT relighting, then Vimeo interpolation. Choose the fourth family after the first three establish whether the same method transfers. Training each task separately can establish a general training recipe; a unified multitask model requires a separate experiment.

| Candidate | Research role | Suggested first dataset | Setup burden | Main limitation |
| --- | --- | --- | --- | --- |
| Controlled relighting | Tests explicit transformation conditions | MIT Multi-Illumination | Medium | A single image and light direction leave hidden scene information uncertain. |
| Deblurring | Practical restoration and fidelity test | GoPro | Low–medium | Blur loses information; a paired target does not establish exact determinism. |
| Frame interpolation | Temporal prediction and motion conditioning | Vimeo-90K triplets | Medium | Occlusions and unseen intermediate motion remain ambiguous. |
| Depth, optionally normals | Dense perception and transfer | Hypersim training; NYUv2/DIODE evaluation | Medium | Close existing one-step methods; geometry is ambiguous from RGB. |
| Physical forward prediction | Stronger test of jointly sufficient inputs | PDEBench 2D Darcy | Medium–high | Must compete against efficient one-pass neural operators. |
| Forward rendering | Closest test of the original proposal | Newly rendered Blender scenes | High | Data generation and matched scene inputs require substantial work. |

The vision tasks assess useful approximate prediction. Fully specified rendering or physical forward problems can additionally test the deterministic-input hypothesis under controlled assumptions.

## Required shared baselines

The closest general comparator is [LBM: Latent Bridge Matching](https://github.com/gojasper/LBM), which already reports single-step image translation across restoration, geometry prediction, and relighting. Its public repository includes checkpoints and a Hypersim normal-training example; its code license is CC BY-NC 4.0. Task breadth alone therefore offers limited distinction from prior work.

Run two complementary comparison tables:

| Table | Comparison | What it establishes |
| --- | --- | --- |
| Matched training | Same task inputs, backbone/pretraining, data split, output representation, and comparable budget | Effect of the source, condition, and objective |
| External methods | Official released methods/checkpoints, with training data, resolution, preprocessing, and runtime recorded | Practical competitiveness against established task solutions |

Within matched training, compare direct residual regression; Gaussian-source conditional FM at 1/4/16/32 evaluations; task-source FM; [MeanFlow](https://arxiv.org/abs/2505.13447) adapted to the same conditional task; and conditional LBM where applicable. Adapted implementations should be labeled as adaptations rather than official paper results. Tune the multi-step reference sufficiently to establish its quality–latency curve.

Every Gaussian-source model receives the same source content through conditioning. Persistent source context, source construction, and condition encoding are part of the design and total latency. A learned initializer additionally needs a comparison against directly decoding its output and against a matched compute prediction network.

## 1. Controlled relighting

### Design

Use an input image under one light as the initial state; condition on source/target lighting descriptors and persistent source-image features; predict the same scene under the requested light. This directly tests whether clear numerical transformation conditions help one-step prediction.

Start with discrete recorded directions, then compare direction IDs against calibrated direction vectors or available light-probe descriptors. Treat interpolation to new continuous light directions as a separate test requiring suitable targets. Measure a fixed-source lighting sweep and split by scene before constructing image pairs.

### Data and access

[MIT Multi-Illumination](https://projects.csail.mit.edu/illumination/) provides more than 1,000 real scenes under 25 lighting conditions, public train/test downloads, and CC-BY data terms. Its advertised JPG training archive is 6.6 GB, making it the practical first dataset; HDR is available separately. These are captured lighting variations, with no ground-truth albedo/G-buffers required for the proposed RGB-to-RGB task.

[GenLit synthetic data](https://huggingface.co/datasets/gbmpi/genlit_dataset/blob/main/README.md) is an optional larger extension with 14-frame lighting sequences and numerical trajectory metadata. The card advertises about 342 GB training and 31 GB test data under an MPI non-commercial research license requiring acceptance and restricting redistribution. Choose MIT first if these terms or download size do not fit the project.

### Baselines

- [ControlNet Single-Image Relighting](https://github.com/graphdeco-inria/controlnet-diffusers-relighting): the 2D component of *A Diffusion Approach to Radiance Field Relighting using Multi-Illumination Synthesis*, with public training/inference code, weights, and MIT-compatible light-direction controls. This is the most directly actionable multi-step comparator.
- Conditional LBM, trained on the same split and lighting descriptors, plus the shared direct/Gaussian-FM/MeanFlow comparisons.
- [GenLit](https://github.com/sbharadwajj/genlit) for lighting sweeps: released inference checkpoints include an MIT mode. Model access requires Hugging Face authentication and acceptance of non-commercial terms. The checked release describes inference rather than a complete matched-training recipe.
- [UniRelight](https://github.com/nv-tlabs/UniRelight) is an additional video-scale comparator when its lighting interface and research license fit the experiment; compare compatible input information and record its external training resources.

### Evaluation

Report image fidelity and perceptual distance, as defined below, across all selected lighting pairs. Hold a source fixed while varying its condition to test controllability; include shadow/specular examples and preserve scene identity. Predefine pair selection so all methods use the same lighting cases.

## 2. Deblurring

### Design

Use the blurry image as source and persistent context; predict its sharp counterpart. First test ordinary paired restoration with available inputs. In a separate controlled experiment, synthesize known blur and supply its kernel or motion parameters to vary condition information; give those parameters to every baseline in that experiment.

The test is practically useful but retains reconstruction ambiguity. Use physical/rendering examples to support stronger claims about uniquely specified targets.

### Data and access

[GoPro](https://seungjunnah.github.io/Datasets/gopro.html) has public downloads under CC BY 4.0. Use the standard 2,103 training / 1,111 test pairs and preserve sequence boundaries. [NAFNet's official preparation guide](https://github.com/megvii-research/NAFNet/blob/main/docs/GoPro.md) supplies a reproducible training/evaluation entry point.

[RealBlur](https://github.com/rimchang/RealBlur) is also released under CC BY 4.0 and provides an official alignment-aware evaluation implementation. Use RealBlur-J as a real-capture transfer test after GoPro training, or train a separate RealBlur model and label the setting clearly. HIDE can be an extra cross-dataset test; this review did not establish its dataset-specific license, so GoPro and RealBlur are the ready first choices.

### Baselines

- [NAFNet](https://github.com/megvii-research/NAFNet) and [Restormer](https://github.com/swz30/Restormer): efficient direct restoration baselines, with established GoPro workflows.
- [DiffIR](https://github.com/Zj-BinXia/DiffIR): diffusion-based restoration with an explicit deblurring implementation.
- [BlurDM](https://github.com/Jin-Ting-He/BlurDM): a more recent Gaussian-noise-conditioned deblurring comparator. Training code is public, but its checked README still lists trained weights as forthcoming; budget retraining before making it mandatory.
- LBM and the shared matched controls. [I2SB](https://github.com/NVlabs/I2SB) is particularly useful for a separate controlled ImageNet-corruption experiment using our existing data; its standard uniform/Gaussian blur setup differs from GoPro motion blur.

### Evaluation

Use the official RGB PSNR/SSIM protocol, perceptual distance, and full generation latency; retain the RealBlur evaluator's alignment/intensity conventions. Measure target VAE reconstruction as a representation reference because its detail loss can confound restoration comparisons; also consider a pixel-space control if latent reconstruction error dominates.

## 3. Frame interpolation

### Design

Given two observed frames and a requested intermediate time tau, predict the intermediate frame. Use either the first frame or a cheaply warped blend as source, and condition on both endpoint frames, tau, and optionally motion estimated solely from those endpoint frames.

Compare the direct initializer output, an equally conditioned residual network, and the proposed transport. Include motion-estimation cost in latency. Separate transport time from video time tau in notation. Ground-truth intermediate-frame flow belongs only in an oracle diagnostic.

### Data and access

[Vimeo-90K](https://github.com/anchen1011/toflow) provides public triplet and septuplet downloads. Start with the triplet benchmark, whose combined archive is advertised as 33 GB. Use septuplets for arbitrary-time experiments, with the official split kept intact.

[SNU-FILM](https://myungsub.github.io/CAIN/) supplies public test triplets grouped into easy/medium/hard/extreme motion settings. Dataset content terms are less explicit on these landing pages than the GoPro release; repository software licenses should not be substituted for rights to every source video. An alternative with clear source terms is to construct and release a documented interpolation split from GoPro's high-frame-rate sequences, labeling it as a new protocol.

SNU-FILM includes GoPro test material, so keep those sequences out of any interpolation training assembled from GoPro.

### Baselines

- [LDMVFI](https://github.com/danier97/LDMVFI): direct multi-step latent-diffusion interpolation comparator; its training resources include Vimeo and BVI-DVC, so use a matched-data retraining or identify released-checkpoint comparisons separately.
- [Consecutive Brownian Bridge](https://github.com/ZonglinL/ConsecutiveBrownianBridge): task-specific bridge-based interpolation.
- [RIFE](https://github.com/hzwer/ECCV2022-RIFE), [AMT](https://github.com/MCG-NKU/AMT), or [GIMM-VFI](https://github.com/GSeanCDAT/GIMM-VFI): strong specialized interpolation comparators. GIMM publishes training/inference code and arbitrary-time benchmarks.
- Shared direct/Gaussian-FM/MeanFlow controls with identical endpoint information.

### Evaluation

Report PSNR/SSIM and perceptual distance by motion difficulty and interpolation time. When generating a sequence of intermediate frames, additionally assess temporal artifacts and include every repeated prediction and motion-estimator call in latency. One renderer evaluation per intermediate frame differs from predicting a whole video in one evaluation.

## 4. Depth and optional normals

### Design

Use RGB-derived latents as source and persistent image features as context, predicting a depth representation. RGB-to-depth source transport is already central to [DepthFM](https://github.com/CompVis/depth-fm); use this as a demanding transfer benchmark rather than a new application claim. Adding sparse measured depth with a validity mask could test stronger conditions, but creates a separate depth-completion task with its own baselines.

### Data and access

[Hypersim](https://github.com/apple-aiml-research/ml-hypersim) releases rendered RGB, depth, normals, and other labels under CC BY-SA 3.0. Train on its official training scenes and download only needed modalities. Access to existing rendered labels is distinct from recreating its artist scenes and renderer pipeline.

For a standard indoor/outdoor synthetic recipe, add [Virtual KITTI 2](https://europe.naverlabs.com/proxy-virtual-worlds-vkitti-2/), whose public files carry CC BY-NC-SA 3.0 and non-commercial terms. If those terms do not fit, use Hypersim-only training and report that setting; outdoor transfer then tests a different training distribution.

Evaluate on the standard [NYUv2](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html) split and [DIODE](https://diode-dataset.org/). NYUv2's labeled release is publicly linked; the checked landing page does not state a clear license. DIODE explicitly releases dataset and code under MIT and provides indoor/outdoor validation data. Use the established evaluation subset/masks rather than treating hidden test ground truth as available. KITTI is optional: official benchmark downloads require login and submissions have eligibility rules; DIODE is a practical first outdoor evaluation without this account dependency.

### Baselines

- [Marigold v1/v1.1](https://github.com/prs-eth/Marigold): multi-step diffusion prediction, with training and evaluation scripts.
- DepthFM and LBM: close one-step structured-source methods; record DepthFM's ensemble size as well as NFE.
- [End-to-end diffusion fine-tuning](https://github.com/RWTHVision/diffusion-e2e-ft): an important one-pass prediction baseline.
- [Marigold V2](https://github.com/huawei-bayerlab/marigold-v2), released in September 2026: a current single-step dense predictor with public training/inference/evaluation code. Its large pretrained backbone makes it useful for external competitiveness; match the backbone for an objective-isolation experiment.
- [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2): a discriminative comparator; its large training resources should be stated in external comparisons. The Small checkpoint uses Apache-2.0, while Base/Large/Giant use CC BY-NC 4.0.

Normals reuse Hypersim and provide another output modality, but should not carry the paper's task-diversity claim by themselves. Add a specialized normal predictor if that task is selected.

### Evaluation

Use the benchmark's depth range, valid masks, crop, and scale/shift alignment consistently. Affine-aligned depth scores and metric-depth scores establish different capabilities; a metric prediction claim requires errors without ground-truth-dependent alignment. Report thin structures and edges alongside overall scores.

## 5. Physical forward prediction

### Design

Start with PDEBench's 2D Darcy forward problem: predict a solution field from its coefficient field under specified forcing and boundary conditions. A first source can be a cheap coarse-grid solution lifted to target resolution; C contains the available coefficient/forcing/boundary information. Also compare coefficient-field initialization and the coarse solution alone.

This is a particularly useful test of abstract task-related sources. Include the coarse solver and any preprocessing in the cost. For positive coefficients and suitable specified boundaries, the deterministic forward setting gives a cleaner target-determination experiment than monocular reconstruction.

### Data and access

[PDEBench's official data record](https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/darus-2986) provides public simulation files under CC BY 4.0, including 2D Darcy and time-dependent PDE families. Download a selected family, pin its dataset version, and split by independent simulation sample rather than individual frames.

### Baselines

- [PDEBench FNO and U-Net implementations](https://github.com/pdebench/PDEBench): mandatory efficient predictors.
- The same coarse numerical solver, and a direct coarse-to-fine residual model.
- Matched conditional Gaussian-source FM/diffusion and MeanFlow, trained on the same fields and conditions. These are proposed adaptations, without directly comparable published Darcy scores established in this review.
- For a later time-dependent PDE experiment, [PDE-Refiner](https://phlippe.github.io/publication/lippe-2023-pderefiner/) and [PDEArena](https://github.com/pdearena/pdearena) supply an iterative-refinement comparison. PDE-Refiner is a rollout method; it is not a drop-in published Darcy comparator or a substitute for the matched Gaussian-source test.

### Evaluation

Report normalized field error, boundary/physical consistency, and total solution time, including the source solve. For dynamics, separately measure single-horizon predictions and repeated-rollout stability, counting all transport evaluations. The scalar-field models need a representation suited to numerical fields; an RGB VAE should not be assumed appropriate.

## 6. Forward rendering alternative

This can replace RGB relighting when the paper emphasizes fully specified physical controls. Generate albedo and cheap shading sources, condition on available geometry/material/light/camera data, and predict a higher-quality reference render.

The inspected [RenderFlow publication page](https://studios.disneyresearch.com/2026/05/31/renderflow-single-step-neural-rendering-via-flow-matching/) provides papers but no verified downloadable training dataset or code release. Its exact experiment should be treated as unavailable for reproduction until access is confirmed. [DiffusionRenderer](https://github.com/nv-tlabs/diffusion-renderer) provides inference code and checkpoints; this does not establish access to its full training set.

Use [BlenderProc](https://github.com/DLR-RM/BlenderProc) or Blender with selected [Poly Haven CC0 assets](https://polyhaven.com/license) and procedural primitives to produce a reproducible new dataset. Export paired scene inputs and targets, and release object-disjoint and lighting-disjoint splits. This is newly generated data, with a rendering budget and asset-selection protocol to establish.

Compare conditional LBM, an implemented RenderFlow-style baseline, DiffusionRenderer where input conventions are compatible, direct residual prediction, matched Gaussian-source FM, and the cheap renderer alone. Label reimplementations and domain/pretraining differences. Screen-space buffers can still omit relevant off-screen transport, so the simulation's fully specified scene and the network's actual observed condition must be distinguished.

## Measurements

| Measure | Purpose, calculation, scale, and reading |
| --- | --- |
| PSNR | Fidelity from mean squared pixel error, expressed as a logarithmic signal/error ratio in decibels; larger is better under identical pixel range and preprocessing. |
| SSIM | Similarity from local luminance/contrast/structure comparisons, dimensionless and usually near 0–1 in these evaluations; larger is better with identical window settings. |
| LPIPS | Learned-feature distance between output and target, dimensionless; smaller indicates greater perceptual similarity under the same network/version. |
| AbsRel | Mean absolute depth error divided by valid reference depth, dimensionless; smaller is better, with alignment declared. |
| Delta-1 | Fraction of valid pixels whose predicted/reference depth ratio is within 1.25 in both directions, reported as fraction or percent; larger is better. |
| Normal angular error | Mean/median angle between predicted and reference unit normals in degrees; smaller is better, using the same camera/world convention and validity mask. |
| Relative L2 field error | Euclidean prediction error divided by reference field norm, dimensionless; smaller is better, with near-zero denominators handled consistently. |
| Total latency | Milliseconds per requested output including preparation, initial prediction/solve, condition extraction, transport, and decoding; smaller is better at comparable quality. |

Use several training seeds for shortlisted matched comparisons. Report single-sample predictions in the main table; sample selection, ensembling, or best-of-N results require their own cost and protocol. External baselines with extra training data or different input information belong in explicitly labeled settings.

## Paper selection

The initial paper candidate is **GoPro/RealBlur deblurring + MIT controlled relighting + Vimeo/SNU-FILM interpolation**. Add **Hypersim-to-real depth** for a vision-focused breadth test, or **PDEBench Darcy** for a broader prediction claim. Forward rendering is a higher-effort alternative to relighting.

Ablate source abstraction and condition informativeness most deeply on relighting or physical forward prediction. Use the other tasks to test whether the recipe transfers without major redesign. Select the benchmark protocol before final evaluations and report the settings where one-step prediction loses quality as well as those where it succeeds.
