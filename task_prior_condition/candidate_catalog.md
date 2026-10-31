# Task candidate catalog

Date: **2026-10-30**. Status: **60 proposals for user selection**. The source/condition choices below are proposed designs. No new task suite, dataset, or backbone has been selected; no downloads or experiments were run.

## Reading the catalog

Each row gives a task, a possible task-related starting sample and inference condition, a benchmark/data route, and task-specific baseline candidates. Starting samples must be represented in the target space or lifted into it; the lift, coarse predictor, solver, or warp contributes to inference cost.

Access labels:

| Label | Meaning |
| --- | --- |
| **Open** | Public release with an explicit data license checked in an official source; attribution and other stated terms still apply. |
| **Public*** | Official project or implementation advertises downloads; complete dataset-specific terms or current archive accessibility remain unverified. |
| **Research** | Explicit academic/non-commercial or research-use terms. |
| **Gated** | Account, access application, signed agreement, or data-use agreement required. |
| **Generate** | Construct paired data or corruptions using permitted inputs; report this as a new protocol. |

These labels concern data access. Model/checkpoint licenses are separate. Named methods are comparison candidates, with no claim that every method has a checkpoint for every proposed split. Where a method uses different inputs or training data, retrain/adapt it or report it in an external-checkpoint setting.

The catalog spans supervised inverse prediction and conditional generation as well as specified forward problems. Test whether the actual inference inputs jointly determine the target before making a deterministic-mapping claim. Appearance synthesis, forecasting, actions, and biological generation also need distribution quality or task-success evaluation when several outcomes are valid.

## Image restoration

| ID / task | Proposed starting sample; condition | Dataset / access | Baseline candidates |
| --- | --- | --- | --- |
| **1. Image denoising** | Noisy RGB; noise level/camera metadata and persistent image features | [SIDD](https://abdokamel.github.io/sidd/) **Open, MIT**; synthetic Gaussian/Poisson noise on licensed images **Generate** | NAFNet; [Restormer](https://github.com/swz30/Restormer); I2SB adaptation |
| **2. Motion deblurring** | Blurry image; source features, optionally available motion cues | [GoPro](https://seungjunnah.github.io/Datasets/gopro.html), [RealBlur](https://github.com/rimchang/RealBlur) **Open, CC BY 4.0** | NAFNet; Restormer; DiffIR; BlurDM; LBM |
| **3. Defocus deblurring** | Defocused image; dual-pixel views or estimated defocus map | [DPDD](https://github.com/Abdullah-Abuolaim/defocus-deblurring-dual-pixel) **Public***; controlled defocus synthesis **Generate** | DPDNet; Restormer single/dual-pixel variants |
| **4. Known-degradation super-resolution** | Upsampled low-resolution image; scale and known kernel | [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) **Research**; degradations on GoPro **Generate** | EDSR; SwinIR; ResShift; [RFMSR](https://github.com/Faze-Hsw/RFMSR) |
| **5. Real-image super-resolution** | Low-quality image or preliminary reconstruction; degradation features | [RealSR](https://github.com/csjcai/RealSR) **Public***; DRealSR via method releases **Public*** | Real-ESRGAN; [DiffBIR](https://github.com/XPixelGroup/DiffBIR); ResShift; RFMSR |
| **6. JPEG artifact removal** | Decoded JPEG; quality factor or estimated compression parameters | DIV2K + fixed JPEG synthesis **Research/Generate**; GoPro-based protocol **Open/Generate**; LIVE1 via baseline loaders **Public*** | [FBCNN](https://github.com/jiaxi-jiang/FBCNN); SwinIR; I2SB; Palette |
| **7. Image inpainting** | Masked image with coarse fill; mask, visible context, optional user hints | [Places2](http://places2.csail.mit.edu/) **Public***; masks on permitted GoPro images **Generate** | LaMa; RePaint; [Palette](https://iterative-refinement.github.io/palette/); LBM |
| **8. Image colorization** | Grayscale image; sparse color hints or palette/reference features | [COCO](https://cocodataset.org/#download) **Public***; existing authorized ImageNet; GoPro conversion **Generate** | [DDColor](https://github.com/piddnad/DDColor); Palette; conditional LBM adaptation |
| **9. Low-light enhancement** | Dark image or inexpensive exposure correction; exposure/noise metadata | LOL-v1/v2 via [Retinexformer](https://github.com/caiyuanhao1998/Retinexformer) **Public***; synthetic exposure on licensed images **Generate** | Retinexformer; LLFlow; direct residual network |
| **10. Image deraining** | Rainy image; source context and rain/degradation descriptor | Rain13K / Rain100H/L via Restormer **Public***; rendered rain **Generate** | Restormer; MPRNet; task-conditioned Gaussian FM adaptation |
| **11. Image dehazing** | Hazy image or preliminary clear estimate; transmission/depth/atmosphere estimates | RESIDE ITS/OTS/SOTS via [DehazeFormer](https://github.com/IDKiro/DehazeFormer) **Public***; synthetic haze with available depth **Generate** | DehazeFormer; direct residual network; conditional Gaussian FM adaptation |
| **12. RAW-to-RGB / learned ISP** | Demosaicked RAW preview; camera metadata, white balance, exposure | Zurich RAW-to-RGB via [PyNET](https://github.com/aiff22/PyNET) **Public***; SIDD RAW + specified pipeline **Open/Generate** | PyNET; conventional ISP; direct residual ISP |

Related variants include mixed-degradation restoration, compression-quality control, fixed-kernel versus blind restoration, and low-light RAW recovery using [SID](https://github.com/cchen156/Learning-to-See-in-the-Dark). These are additional protocols within the families above, rather than automatically independent evidence of task diversity.

## Image formation and controlled synthesis

| ID / task | Proposed starting sample; condition | Dataset / access | Baseline candidates |
| --- | --- | --- | --- |
| **13. Controlled relighting** | Image under source light; target light descriptor and source appearance context | [MIT Multi-Illumination](https://projects.csail.mit.edu/illumination/) **Open, CC-BY**; GenLit **Research/Gated terms** | LBM; [ControlNet relighting](https://github.com/graphdeco-inria/controlnet-diffusers-relighting); GenLit; UniRelight |
| **14. Forward neural rendering** | Albedo or cheap shaded render; geometry, material, camera, lighting | Blender/BlenderProc + [Poly Haven](https://polyhaven.com/license) CC0 assets **Generate** | RenderFlow-style reimplementation; DiffusionRenderer; LBM; direct renderer |
| **15. Intrinsic decomposition** | RGB-derived state; image context and requested intrinsic modality | [Hypersim](https://github.com/apple-aiml-research/ml-hypersim) **Open, CC BY-SA 3.0** | Marigold intrinsic models; [albedoLBM](https://github.com/CVC-Color/albedoLBM); direct decomposition network |
| **16. Image harmonization** | Composite image; foreground mask and background appearance | [iHarmony4](https://github.com/bcmi/Image-Harmonization-Dataset-iHarmony4) **Public***; controlled composites **Generate** | DoveNet; [Harmonizer](https://github.com/SitongLiao/Harmonizer); conditional LBM adaptation |
| **17. Shadow removal** | Shadowed image; observed/estimated shadow mask and image context | ISTD/ISTD+/SRD through [ShadowFormer](https://github.com/GuoLanqing/ShadowFormer) **Public***; rendered shadows **Generate** | ShadowFormer; direct residual model; conditional bridge adaptation |
| **18. Photo retouching** | Original or globally corrected photo; requested expert/style code | [MIT-Adobe FiveK](https://data.csail.mit.edu/graphics/fivek/) **Research, per-file license groups**; controlled edit programs **Generate** | HDRNet; Harmonizer/Enhancer; direct residual retoucher |
| **19. Multi-exposure HDR reconstruction** | Simple exposure merge; exposure stack and camera/exposure settings | Kalantari HDR dataset through [HDR-Transformer](https://github.com/megvii-research/HDR-Transformer) **Public***; controlled rendered brackets **Generate** | HDR-Transformer; AHDRNet; conventional HDR merge |
| **20. Virtual try-on** | Warped garment/person composite; pose, garment image, agnostic mask | [Dress Code](https://github.com/aimagelab/dress-code) **Gated, academic restrictions**; VITON-HD under its own terms | [IDM-VTON](https://github.com/yisol/IDM-VTON); StableVITON; HR-VITON |
| **21. Pose-guided person synthesis** | Coarsely warped person; target pose and appearance reference | [DeepFashion In-shop](https://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/InShopRetrieval.html) **terms/access to confirm**; consented rendered avatar pairs **Generate** | [PIDM](https://github.com/ankanbhunia/PIDM); NTED; conditional bridge adaptation |
| **22. Semantic-layout-to-image synthesis** | Colored layout or coarse render; semantic layout and appearance/style reference | ADE20K **Gated full release** / scene-parsing subset; COCO-Stuff **Public***; Kubric layouts **Generate** | [SPADE](https://github.com/NVlabs/SPADE); ControlNet; BBDM/DDBM adaptation |

For source abstraction, these tasks offer controllable comparisons such as albedo versus cheap shading, composite versus foreground-only reference, and semantic layout versus an appearance-bearing image. The condition can restore selected information at controlled capacity.

## Dense perception

| ID / task | Proposed starting sample; condition | Dataset / access | Baseline candidates |
| --- | --- | --- | --- |
| **23. Monocular depth** | RGB-derived latent or coarse depth; image features and camera intrinsics where available | Hypersim **Open**; [DIODE](https://diode-dataset.org/) **Open, MIT**; NYUv2 **Public***; vKITTI2 **Research** | DepthFM; Marigold v1/v2; LBM; Depth Anything V2; E2E-FT |
| **24. Surface normals** | RGB-derived latent or coarse normal estimate; RGB and camera context | Hypersim / DIODE **Open**; ScanNet **Gated** if added | [DSINE](https://github.com/baegwangbin/DSINE); LBM; Marigold normals; E2E-FT |
| **25. Semantic segmentation** | Coarse logits; RGB features and optional sparse labels | [ADE20K](https://groups.csail.mit.edu/vision/ADE20K/index.html) **registration for full release**; COCO-Stuff **Public***; synthetic labels **Generate** | [Mask2Former](https://github.com/facebookresearch/Mask2Former); SegFormer; Gaussian conditional FM adaptation |
| **26. Depth completion** | Interpolated sparse depth; sparse measurements, validity mask, RGB | KITTI depth completion **Gated**; NYUv2 **Public***; sparse DIODE/Hypersim **Open/Generate** | [CompletionFormer](https://github.com/youmi-zym/CompletionFormer); NLSPN; Marigold V2 completion |
| **27. Optical flow** | Coarse flow estimate; two RGB frames and image features | Sintel / FlyingChairs / FlyingThings3D **Public*, respective terms**; [Kubric](https://github.com/google-research/kubric) **Generate** | RAFT; FlowFormer; [FlowDiffuser](https://github.com/LA30/FlowDiffuser) |
| **28. Stereo matching** | Coarse disparity; rectified stereo pair, calibration, cost-volume features | Scene Flow **Public***; KITTI stereo **Gated**; vKITTI2 **Research** | [RAFT-Stereo](https://github.com/princeton-vl/RAFT-Stereo); IGEV-Stereo; conditional Gaussian FM adaptation |
| **29. Image matting** | Trimap-derived coarse alpha; RGB and trimap | Composition-1k **access/terms to confirm**; other released sets through [ViTMatte](https://github.com/hustvl/ViTMatte) **Public***; synthetic alpha composites **Generate** | ViTMatte; [DiffMatte](https://github.com/YihanHu-2022/DiffMatte); direct alpha predictor |
| **30. Instance/panoptic segmentation** | Coarse mask proposals; image features and class/query context | COCO **Public***; synthetic object masks **Generate** | Mask2Former; Mask R-CNN; conditional continuous-mask FM adaptation |

Coarse predictions must come from inference-available inputs. Compare directly against the coarse predictor and a residual refiner with similar total computation. Segmentation additionally needs a defined continuous mask/logit representation and decoding rule; flow matching on arbitrary integer class IDs would introduce an artificial class ordering.

## Video, views, and motion

| ID / task | Proposed starting sample; condition | Dataset / access | Baseline candidates |
| --- | --- | --- | --- |
| **31. Video frame interpolation** | Endpoint frame or warped blend; two endpoint frames and requested video time | [Vimeo-90K](https://github.com/anchen1011/toflow), SNU-FILM **Public***; GoPro-based split **Open/Generate** | LDMVFI; Consecutive Brownian Bridge; RIFE; GIMM-VFI |
| **32. Video super-resolution** | Upsampled low-res clip; neighboring frames and scale/degradation | REDS / Vimeo-90K via [RVRT](https://github.com/JingyunLiang/RVRT) **Public*** | BasicVSR++; RVRT; VRT; conditional Gaussian video FM adaptation |
| **33. Video deblurring** | Blurry clip; adjacent frames and temporal alignment features | DVD / GoPro-derived clips via RVRT **Public***; REDS **Public*** | RVRT; VRT; framewise Restormer |
| **34. Video denoising** | Noisy clip; noise level and adjacent frames | DAVIS / Set8 via [VRT](https://github.com/JingyunLiang/VRT) **Public***; synthetic noise **Generate** | FastDVDnet; RVRT; VRT |
| **35. Short-horizon video prediction** | Last observed frame or coarse rollout; observed history and available actions | BAIR via [MCVD](https://github.com/voletiv/mcvd-pytorch) **Public***; Kubric controlled rollouts **Generate** | MCVD; direct video predictor; state-conditioned renderer adaptation |
| **36. Novel-view synthesis** | Warped/splatted source views; source images, camera poses, target camera | RealEstate10K through [pixelSplat](https://github.com/dcharatan/pixelsplat) **Public*, source-video availability varies**; DTU **Public***; synthetic views **Generate** | pixelSplat; MVSplat; FlowR; Gaussian-source view synthesis adaptation |
| **37. Human motion in-betweening** | Interpolated skeleton sequence; keyframes, timestamps, root trajectory | HumanML3D/AMASS **Gated / research and upstream terms**; procedural skeleton motion **Generate** | [CondMDI](https://github.com/setarehc/diffusion-motion-inbetweening); MDM; direct motion interpolator |

These rows include distinct temporal protocols. Generating one frame in one evaluation and generating an entire clip in one evaluation require separate compute accounting. For richer physical language, a Kubric variant can supply object states/events as explicit conditions and compare supplied versus predicted state sequences.

## 3D geometry

| ID / task | Proposed starting sample; condition | Dataset / access | Baseline candidates |
| --- | --- | --- | --- |
| **38. Point-cloud upsampling** | Interpolated/duplicated sparse points; local geometry and upsampling factor | PU1K / PU-GAN through [PUFM](https://github.com/Holmes-Alan/PUFM) **Public*, shape-source terms to check**; CC0 mesh sampling **Generate** | PUFM; PU-GCN; Grad-PU |
| **39. Point-cloud completion** | Partial cloud plus coarse completion; observed points, view/visibility features | PCN / ShapeNet-55 through [PoinTr](https://github.com/yuxumin/PoinTr) **upstream ShapeNet terms/Gated**; CC0 meshes **Generate** | PoinTr; AdaPoinTr; PCN; Gaussian conditional point-flow adaptation |
| **40. Point-cloud denoising** | Noisy points; local neighborhoods and noise descriptor | PU-Net evaluation data through [Score-Denoise](https://github.com/luost26/score-denoise) **Public***; CC0 mesh sampling + noise **Generate** | Score-Denoise; StraightPCF; PointCleanNet |
| **41. LiDAR scene completion** | Sparse voxel/range representation; scan, calibration, visibility context | [SemanticKITTI](https://www.semantic-kitti.org/dataset.html) **Research, CC BY-NC-SA; underlying KITTI access**; simulated scans **Generate** | LMSCNet; SSCNet; conditional Gaussian occupancy/point-flow adaptation |

Point ordering/coupling and representation size require a deliberate design. A dense geometry target is a set, so compare geometric quality with permutation-invariant metrics rather than treating an arbitrary point order as a uniquely specified target.

## Physical and geophysical prediction

| ID / task | Proposed starting sample; condition | Dataset / access | Baseline candidates |
| --- | --- | --- | --- |
| **42. Darcy forward solution** | Coarse pressure solution; permeability, forcing, boundaries | [PDEBench](https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/darus-2986) **Open, CC BY 4.0** | FNO; U-Net; coarse solver; direct residual solver; Gaussian conditional FM |
| **43. Fluid evolution** | Previous field or coarse rollout; physical parameters, forcing, horizon | PDEBench **Open**; PDEArena **Public*, selected data terms** | FNO; U-Net; [PDE-Refiner](https://phlippe.github.io/publication/lippe-2023-pderefiner/); ACDM |
| **44. Elasticity / stress fields** | Coarse elastic response; geometry, material, loading/boundaries | [Geo-FNO elasticity](https://github.com/neuraloperator/Geo-FNO) **Public***; finite-element simulation **Generate** | Geo-FNO; DeepONet adaptation; coarse FEM; direct residual predictor |
| **45. Cloth / deformable simulation** | Current mesh or inexpensive integration; recent states, forces, constraints | [MeshGraphNets flag data](https://github.com/google-deepmind/deepmind-research/tree/master/meshgraphnets) **Public***; own mesh simulation **Generate** | MeshGraphNets; direct graph predictor; coarse integrator |
| **46. Weather forecasting** | Persistence/coarse forecast; recent atmosphere, static fields, horizon | [WeatherBench 2](https://weatherbench2.readthedocs.io/en/latest/data-guide.html) **public bucket, per-dataset terms** | GraphCast; Pangu-Weather; GenCast; matched-resolution conditional FM |
| **47. Weather downscaling** | Interpolated coarse weather field; coarse variables and terrain | ERA5/CERRA via Copernicus **account/terms**; WeatherBench-derived coarse-to-fine protocol **per-dataset terms/Generate** | [CorrDiff](https://github.com/NVIDIA/physicsnemo/tree/main/examples/weather/corrdiff) adaptation; deterministic U-Net; residual downscaler |
| **48. Seismic inversion** | Smooth/coarse velocity model; seismic traces and acquisition geometry | [OpenFWI](https://openfwi-lanl.github.io/) **Public*, data license to check separately from code**; acoustic simulation **Generate** | InversionNet; VelocityGAN; classical FWI; conditional Gaussian FM adaptation |

Fully specified well-posed forward problems provide a clean setting for the target-determination hypothesis. Coarse states, unresolved forcing, inverse measurements, and chaotic long-horizon dynamics can reintroduce uncertainty. Weather foundation models use substantial resources; compare matched resolution/data and separately label external checkpoints. CorrDiff is an adaptation candidate for a new regional protocol, not a source of ready-made scores on arbitrary ERA5/CERRA pairs.

## Audio and speech

| ID / task | Proposed starting sample; condition | Dataset / access | Baseline candidates |
| --- | --- | --- | --- |
| **49. Speech enhancement** | Noisy waveform/STFT; noisy context, optional noise features | VoiceBank-DEMAND **Public***; [LibriSpeech](https://www.openslr.org/12/) CC BY 4.0 + [SLR28 noise](https://www.openslr.org/28/) Apache-2.0 **Open/Generate** | [SGMSE](https://github.com/sp-uhh/sgmse); StoRM; Li et al. [MeanFlowSE](https://github.com/liduojia1/MeanFlowSE); discriminative enhancer |
| **50. Speech dereverberation** | Reverberant audio or coarse dereverb output; reverberant context and available room cues | LibriSpeech + SLR28 impulse responses **Open/Generate**; REVERB's speech-source access separately | SGMSE dereverberation; WPE; direct waveform/spectrogram predictor |
| **51. Audio bandwidth extension** | Resampled low-bandwidth audio; input bandwidth and audio context | [VCTK 0.92](https://datashare.ed.ac.uk/handle/10283/3443) **Open, CC BY 4.0**; LibriSpeech **Open** with synthesized bandwidth loss | [AudioSR](https://github.com/haoheliu/versatile_audio_super_resolution); NU-Wave 2; direct bandwidth extender |
| **52. Speech source separation** | Mixture-derived source guesses; mixture and optional target-speaker reference | [LibriMix](https://github.com/JorisCos/LibriMix) speech-only mixes from LibriSpeech **Open/Generate**; noisy variants additionally require WHAM-noise terms | SepFormer; Conv-TasNet; conditional diffusion separator adaptation |
| **53. Neural vocoding** | Griffin-Lim/coarse waveform; mel spectrogram and optional speaker features | [LJ Speech](https://keithito.com/LJ-Speech-Dataset/) **Open, public domain**; VCTK **Open** | [HiFi-GAN](https://github.com/jik876/hifi-gan); [BigVGAN](https://github.com/NVIDIA/BigVGAN); DiffWave; WaveGrad |

The speech enhancement literature already contains close structured-source and one-step methods. Distinguish the two papers named MeanFlowSE; the linked row refers to Li et al.'s conditional-mean-flow implementation. Recovered audio can be plausible without matching a unique waveform phase, so include perceptual and intelligibility measures alongside paired errors.

## Medical and scientific imaging

| ID / task | Proposed starting sample; condition | Dataset / access | Baseline candidates |
| --- | --- | --- | --- |
| **54. Accelerated MRI reconstruction** | Zero-filled or coarse reconstructed image; acquired k-space, mask, coil information | [fastMRI](https://fastmri.med.nyu.edu/) **Gated, institutional application/DUA**; simulated single-coil acquisition on IXI **Open/Generate, different protocol** | [fastMRI VarNet](https://github.com/facebookresearch/fastMRI); U-Net; score-based MRI reconstruction; Gaussian conditional FM |
| **55. CT reconstruction** | Filtered backprojection; sinogram and measurement geometry/noise | [LoDoPaB-CT](https://zenodo.org/records/3384092) **public release**; custom phantoms **Generate** | FBP + U-Net; learned primal-dual; TV reconstruction; DPS adaptation |
| **56. MRI contrast synthesis** | T1-derived state; source contrast and requested target contrast | [IXI](https://brain-development.org/ixi-dataset/) **Open, CC BY-SA 3.0**; registration/preprocessing needed | [pGAN/cGAN](https://github.com/icon-lab/pGAN-cGAN); paired U-Net; conditional bridge adaptation |
| **57. Microscopy denoising** | Noisy fluorescence frame; noise level and available repeated observations | [FMD](https://github.com/yinhaoz/denoising-fluorescence) **Public***; synthetic Poisson-Gaussian corruption **Generate** | CARE; Noise2Void; FMD baseline; matched Gaussian conditional FM |

Respect subject-level separation and measurement operators. IXI-derived simulated MRI acquisition is a useful alternative when fastMRI access is unavailable, with a clearly different task protocol. Contrast synthesis can have genuinely missing pathological information; evaluate it as conditional synthesis rather than assuming source contrast determines all target details.

## Actions, planning, and biological structures

| ID / task | Proposed starting sample; condition | Dataset / access | Baseline candidates |
| --- | --- | --- | --- |
| **58. Robot action-chunk prediction** | Warm-start or coarse action sequence; observations, proprioception, goal | [robomimic](https://robomimic.github.io/docs/datasets/overview.html) / [Push-T](https://github.com/real-stanford/diffusion_policy) **public routes; data/simulator terms** | Diffusion Policy; behavior cloning; conditional MeanFlow policy adaptation |
| **59. Goal-conditioned trajectory planning** | Straight-line/coarse path; start, goal, obstacles or state constraints | Maze2D/D4RL via [Diffuser](https://github.com/jannerm/diffuser) **public routes; environment/version terms**; procedural 2D mazes **Generate** | Diffuser; DiffuserLite; A*/trajectory optimization; direct planner |
| **60. Protein motif scaffolding** | Coarse/partially specified backbone; motif coordinates and structural constraints | [PDB archive](https://www.rcsb.org/pages/usage-policy) **Open, CC0 archive data**; curated motif splits | [FrameFlow](https://github.com/microsoft/protein-frame-flow); RFdiffusion; conditional structural-flow adaptation |

These tasks intentionally allow multiple valid outputs. Judge success, constraints, coverage, and generation cost; a single paired target error cannot establish their quality. Their domain-specific representations and evaluation pipelines make them substantial separate research efforts.

## Access alternatives

| Restricted or uncertain route | Alternative route | Protocol implication |
| --- | --- | --- |
| DIV2K's academic-use terms | GoPro clean images or self-rendered images for controlled corruption | New restoration training/evaluation protocol; report it explicitly. |
| Places2/ImageNet access or terms unsuitable | Masks/color conversion on permitted images, or rendered scenes | Same task family; different distribution from the named benchmark. |
| RenderFlow's unverified training-data release | BlenderProc + procedural geometry and Poly Haven CC0 assets | New rendering benchmark and generation budget. |
| GenLit terms/download size unsuitable | MIT Multi-Illumination | Real RGB relighting with discrete recorded lighting. |
| Dress Code application not available | VITON-HD only after separately verifying its terms; otherwise consented/self-rendered garment-person data | No automatic unrestricted substitute for the Dress Code benchmark. |
| DeepFashion terms unavailable | Consented or rendered avatar/pose pairs | New pose-transfer protocol. |
| ADE20K full-release registration | Check the separately offered scene-parsing release, or use COCO/synthetic segmentation with their own terms | Change of benchmark or release must be named. |
| KITTI/vKITTI restrictions | DIODE/Hypersim with simulated sparse depth or Kubric stereo/flow | New sensing protocol; DIODE is not a replacement for real driving stereo. |
| Composition-1k access unresolved | Rendered alpha composites or another matting release after checking its data terms | New matting protocol and foreground distribution. |
| HumanML3D/AMASS/SMPL access unsuitable | Procedural articulated motion or separately permitted motion capture | Different motion benchmark and possibly representation. |
| ShapeNet upstream gating | CC0 meshes and procedural shapes for partial/dense/noisy sampling | New shape-disjoint geometry benchmark. |
| SemanticKITTI/KITTI terms unsuitable | Simulated LiDAR from permitted scenes | New sensor/domain benchmark. |
| Weather data account/terms unsuitable | A specifically permitted public WeatherBench subset; PDEBench for physical prediction | WeatherBench subsets still retain their data terms; PDEBench changes the task. |
| REVERB/WSJ speech-source access unavailable | LibriSpeech + public SLR28 impulse responses | Reproducible synthetic dereverberation protocol. |
| fastMRI application unavailable | IXI + simulated Fourier undersampling | Simulated single-coil reconstruction rather than the fastMRI challenge. |

Official access details worth retaining: [Dress Code](https://github.com/aimagelab/dress-code) requires an institutional address and signed application and excludes private companies; [ShapeNet](https://huggingface.co/ShapeNet) now distributes through dataset-specific Hugging Face access requests; [fastMRI](https://fastmri.med.nyu.edu/) requires an application, institutional affiliation, and data agreement. A public mirror does not replace the original data terms.

## Shared comparisons after selection

For selected tasks, compare a direct predictor/residual predictor, Gaussian-source conditional FM over a quality–latency curve, the task-source method, and an appropriate dedicated one-step method. [MeanFlow](https://arxiv.org/abs/2505.13447) and [LBM](https://github.com/gojasper/LBM) are useful reference frameworks, with adaptations labeled. Every method receives the same inference-available information in the controlled comparison.

If C adds information, measure that contribution separately from changing the source. If C reorganizes information already present in S, study routing and model access. If a learned initializer supplies most of the target, compare against it directly and count its computation.

The user's selections determine the next detailed benchmark plans. Stable IDs 1–60 are provided for referring to candidates; the catalog does not select a paper suite.
