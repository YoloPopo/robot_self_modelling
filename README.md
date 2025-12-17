# Real-Time Robot Self-Modeling via Direct Sensorimotor Decoding

**Course:** Machine Learning and Data Mining Implementation Project
**Institution:** Higher School of Economics
**Author:** Muhammad Zeeshan Asghar

## Project Overview

This project investigates the "Reality Gap" in robotics—the discrepancy between a robot's internal kinematic model (URDF) and its physical state, often caused by damage or mechanical drift. To address this, we implement and compare three distinct self-modeling architectures:

1.  **FFKSM (Baseline):** A reproduction of the Neural Radiance Field (NeRF) approach from *"Teaching Robots to Build Simulations of Themselves"* (Nature Machine Intelligence, 2024).
2.  **K-3DGS (Hypothesis):** An experimental Kinematic 3D Gaussian Splatting approach aimed at reducing rendering latency.
3.  **NeuroKin (Innovation):** A novel, direct sensorimotor decoding architecture proposed to achieve sub-millisecond inference speeds for real-time control loops.

## Key Results (Summary)

| Architecture | Approach | FPS | Latency | PSNR (Quality) | Outcome |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **FFKSM** | NeRF / Volumetric Ray Marching | 5.22 | 192 ms | 17.35 dB | **Too Slow** (Failed Latency Req) |
| **K-3DGS** | Kinematic 3D Gaussians | 37 | 27 ms | 17.01 dB | **Visual Artifacts** (Blobby) |
| **NeuroKin** | Direct Sensorimotor Decoding | **7,400** | **0.13 ms** | **21.88 dB** | **SOTA Performance** |
| **ResNeuroKin-D**| Multi-Task + Heuristic Depth | 2,500 | 0.40 ms | 21.23 dB | **Geometric Robustness** |

---

## 1. Data Generation Pipeline

**Notebook:** `1. data_generation.ipynb`

To overcome the unavailability of the original paper's dataset, a custom data generation pipeline was built using **PyBullet**.

* **Algorithmic Approach:** Instead of random motor babbling (which produces jerky, non-ergodic motion), we implemented **Lorenz Attractor trajectories**. This chaotic but deterministic system ensures smooth, continuous exploration of the robot's workspace.
* **Dataset Specifications:**
    * **Input:** 4-DOF Joint Angles.
    * **Target:** $100 \times 100$ RGB Images (processed to binary masks).
    * **Scale:** 2,000 Total Samples (1,600 Train / 400 Test). *Note: This constrained dataset size serves as a stress test for model data efficiency.*

---

## 2. Methodologies & Implementation Details

### A. Baseline: Free Form Kinematic Self-Model (FFKSM)
**Notebook:** `2. NERF.ipynb`

We reproduced the FFKSM architecture, which utilizes a **Split-Encoder NeRF**. Ideally, this allows the robot to learn its shape by querying a neural field conditioned on joint angles.

* **Virtual Frame Prior:** To simplify learning, the 3D coordinate system is rigidly transformed by the base joints ($A_0, A_1$) before entering the network.
* **Engineering Challenges & Fixes:**
    * **"Black Screen" Convergence:** The model initially converged to a local minimum of outputting all-black images due to class imbalance (the robot is thin). **Fix:** Implemented a **Center-Cropping Curriculum**, forcing the model to train only on the center $50 \times 50$ pixels for the first 500 iterations.
    * **Kinematic Blindness:** Fixed a tensor slicing bug where the network received only partial joint states ($A_0, A_1$), ignoring the end-effector.
* **Critique:** While the model successfully reconstructed the robot, the **Ray Marching** process (sampling 64 points per pixel) capped performance at **5.22 FPS**, making it non-viable for real-time control.

### B. Hypothesis: Kinematic 3D Gaussian Splatting (K-3DGS)
**Notebook:** `3. K-3DGS.ipynb`

We attempted to solve the NeRF latency bottleneck by attaching **3D Gaussian primitives** to the robot links and using rasterization (Splatting) instead of ray marching.

* **Failure Analysis:** The optimization proved unstable when using anisotropic (ellipsoidal) Gaussians for thin kinematic links, leading to exploding gradients on rotation matrices.
* **Compromise:** We were forced to use **Isotropic (Spherical) Gaussians** to stabilize training.
* **Result:** While faster (~37 FPS), the spherical primitives could not model the thin robot arms accurately, resulting in "blobby," disconnected visual artifacts and low PSNR (17.01 dB).

### C. Innovation: NeuroKin & ResNeuroKin-D
**Notebooks:** `4. NeuroKin_Motor_Visual_Decoder.ipynb`, `5. ResNeuroKinD_Dual_Head_Depth.ipynb`

Recognizing that 3D reconstruction is computationally expensive and unnecessary for simple collision checking, we proposed **NeuroKin**.

* **Approach:** Direct Sensorimotor Decoding. An Encoder-Decoder network maps Joint Angles directly to Visual Masks in a single forward pass ($O(1)$ complexity).
* **ResNeuroKin-D:** An extension that adds a secondary **Depth Head**. Since no depth sensor was available, we trained this head on **Heuristic Depth** (synthetic kinematic distance). This forces the latent space to organize geometrically, preventing the model from merely memorizing 2D textures.
* **Result:** Achieved **7,400 FPS**, a 1,400x speedup over the FFKSM baseline.

---

## Repository Structure

### Core Implementation
* `1. data_generation.ipynb`: PyBullet simulation setup using Lorenz Attractors to generate the dataset.
* `2. NERF.ipynb`: Implementation of the FFKSM Baseline (NeRF).
* `3. K-3DGS.ipynb`: Implementation of Kinematic 3D Gaussian Splatting.
* `4. NeuroKin_Motor_Visual_Decoder.ipynb`: The high-speed direct decoding model.
* `5. ResNeuroKinD_Dual_Head_Depth.ipynb`: Multi-task learning model with heuristic depth estimation.

### Data & Assets
* `data/`: Contains simulation data, actions, and training outputs (logs/models).
* `RobotArmURDF/`: Contains the URDF description files and meshes for the 4-DOF manipulator used in PyBullet.
* `figures/`: Generated plots and qualitative comparison images.

### Documentation
* `Report.pdf`: Full academic report detailing the theoretical background and experiments.
* `Presentation.pdf`: Defense slides summarizing the project.
* `Reference Papers/`: Collection of literature reviewed for this implementation.

---

## Installation & Usage

### Prerequisites
The project requires a Python environment with PyTorch and PyBullet.

```bash
pip install torch numpy opencv-python matplotlib pybullet tqdm