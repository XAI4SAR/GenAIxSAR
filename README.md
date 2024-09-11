# GenAI Meets SAR: A List of Resources

## Awesome papers

### Review & Survey Papers

**GenAI Technology**

[A Survey on Generative Modeling with Limited Data, Few Shots, and Zero Shot](http://arxiv.org/abs/2307.14397)

[Controllable Data Generation by Deep Learning: A Review](https://arxiv.org/pdf/2207.09542)

[Diffusion Models: A Comprehensive Survey of Methods and Applications](https://arxiv.org/pdf/2209.00796)

Making Images Real Again: A Comprehensive Survey on Deep Image Composition

**Synthetic Aperture Radar**

Application of deep generative networks for SAR/ISAR: a review

A review of Generative Adversarial Networks (GANs) and its applications in a wide variety of disciplines - From Medical to Remote Sensing

Deep Learning Methods For Synthetic Aperture Radar Image Despeckling: An Overview Of Trends And Perspectives

Explainable, Physics-Aware, Trustworthy Artificial Intelligence: A paradigm shift for synthetic aperture radar

微波视觉与SAR图像智能解译

A review and meta-analysis of Generative Adversarial Networks and their applications in remote sensing

### Electromagnetic Modeling

SARCASTIC v2.0—High-Performance SAR Simulation for Next-Generation ATR Systems

[Ray-Tracing Simulation Techniques for Understanding High-Resolution SAR Images](http://ieeexplore.ieee.org/document/5238514/)

Potentials and limitations of SAR image simulators – A comparative study of three simulation approaches

[Imaging Simulation of Polarimetric SAR for a Comprehensive Terrain Scene Using the Mapping and Projection Algorithm](https://ieeexplore.ieee.org/document/1717717)

[RaySAR - 3D SAR simulator: Now open source](http://ieeexplore.ieee.org/document/7730757/)

### Statistic Modeling

[Numerical Simulation of SAR Image for Sea Surface](https://www.mdpi.com/2072-4292/14/3/439)

[Synthetic Aperture Radar Image Statistical Modeling: Part One-Single-Pixel Statistical Models](https://ieeexplore.ieee.org/document/9166719/)

[Synthetic Aperture Radar Image Statistical Modeling: Part Two-Spatial Correlation Models and Simulation](https://ieeexplore.ieee.org/document/9262904/)

[A Facet-Based Numerical Model for Simulating SAR Altimeter Echoes From Heterogeneous Sea Ice Surfaces](https://ieeexplore.ieee.org/document/8625441/)

[Statistical Modeling of Polarimetric SAR Data: A Survey and Challenges](http://www.mdpi.com/2072-4292/9/4/348)

[A Physical Analysis of Polarimetric SAR Data Statistical Models](http://ieeexplore.ieee.org/document/7377073/)

### Physics-Inspired GenAI Methods

**NeRF + Radar:**

Radar Fields: An Extension of Radiance Fields to SAR

DART: Implicit Doppler Tomography for Radar Novel View Synthesis

[Radar Fields: Frequency-Space Neural Scene Representations for FMCW Radar](https://dl.acm.org/doi/10.1145/3641519.3657510)

[ISAR-NeRF: Neural Radiance Fields for 3-D Imaging of Space Target From Multiview ISAR Images](https://ieeexplore.ieee.org/document/10423594/?arnumber=10423594)

Circular SAR Incoherent 3D Imaging with a NeRF-Inspired Method

[RaNeRF: Neural 3-D Reconstruction of Space Targets From ISAR Image Sequences](https://ieeexplore.ieee.org/document/10190736/?arnumber=10190736)

**Physics Meets GenAI in computer vision:**

[PAC-NeRF: Physics Augmented Continuum Neural Radiance Fields for Geometry-Agnostic System Identification](http://arxiv.org/abs/2303.05512)

[Physics-Informed Guided Disentanglement in Generative Networks](https://ieeexplore.ieee.org/document/10070869/)

[Model-Based Deep Learning](https://ieeexplore.ieee.org/document/10056957/)

[PhyRecon: Physically Plausible Neural Scene Reconstruction](http://arxiv.org/abs/2404.16666)

[Physically-aware Generative Network for 3D Shape Modeling](https://ieeexplore.ieee.org/document/9578796/)

### AI-Empowered Physical Model

[Dynamic ocean inverse modeling based on differentiable rendering](https://link.springer.com/10.1007/s41095-023-0338-4)

[Differentiable Rendering for Synthetic Aperture Radar Imagery](https://ieeexplore.ieee.org/document/10214298/)

[Learning Surface Scattering Parameters From SAR Images Using Differentiable Ray Tracing](http://arxiv.org/abs/2401.01175)

[Reinforcement Learning for SAR View Angle Inversion with Differentiable SAR Renderer](http://arxiv.org/abs/2401.01165)

[Extension of Differentiable SAR Renderer for Ground Target Reconstruction From Multiview Images and Shadows](https://ieeexplore.ieee.org/document/10266368/)

[Differentiable SAR Renderer and Image-Based Target Reconstruction](https://ieeexplore.ieee.org/document/9926979/)

[Model-Based Information Extraction From SAR Images Using Deep Learning](https://ieeexplore.ieee.org/document/9286839/)

[A SAR Target Image Simulation Method With DNN Embedded to Calculate Electromagnetic Reflection](https://ieeexplore.ieee.org/abstract/document/9345961)

[Parameter Extraction Based on Deep Neural Network for SAR Target Simulation](https://ieeexplore.ieee.org/document/8999587/)

## Datasets

### Multi-view SAR Target Generation

The moving and stationary target acquisition and recognition (MSTAR) dataset

SAMPLE dataset

The Synthetic and Measured Paired and Labeled Experiment (SAMPLE) dataset

[飞机目标多角度SAR数据集](https://radars.ac.cn/web/data/getData?newsColumnId=1c9a6287-4763-4f94-889e-156f50aca946)

OpenSARShip dataset

FUSARShip dataset

### SAR-to-Optical Image Translation

SEN1-2: The SEN1-2 Dataset for Deep Learning in SAR-Optical Data Fusion

SAR2Opt: A Comparative Analysis of GAN-Based Methods for SAR-to-Optical Image Translation

QXS-SAROPT: The QXS-SAROPT Dataset for Deep Learning in SAR-Optical Data Fusion

SEN12MS: SEN12MS – A Curated Dataset of Georeferenced Multi-Spectral Sentinel-1/2 Imagery for Deep Learning and Data Fusion

WHU-SEN-City: SAR-to-Optical Image Translation Using Supervised Cycle-Consistent Adversarial Networks

Multi-Sensor All Weather Mapping (MSAW) Dataset: SpaceNet 6: Multi-Sensor All Weather Mapping Dataset

## Experiments

We provide several baseline models based on GAN for multi-view SAR target image generation under limited observation angles. The source code can be found at ./GAN

