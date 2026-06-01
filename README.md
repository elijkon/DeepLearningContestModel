# 🚀 Latent Flow Matching Generative Pipeline
> **A High-Performance, Dimensionally-Compressed Deep Generative AI Model**
> *Developed by Elijah Konkle & Evan Dant | October 2025*

[![Google Colab](https://img.shields.io/badge/Google%20Colab-Active-orange?logo=googlecolab&logoColor=white)](https://colab.research.google.com/github/elijkon/DeepLearningContestModel/blob/main/DeepLearningContest.ipynb)
[![Framework PyTorch](https://img.shields.io/badge/Framework-PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Experiment Tracking Weights & Biases](https://img.shields.io/badge/Tracking-Weights%20%26%20Biases-FFBE00?logo=weightsandbiases&logoColor=black)](https://wandb.ai/)
[![Model Format Safetensors](https://img.shields.io/badge/Format-Safetensors-0080FF?logo=huggingface&logoColor=white)](https://github.com/huggingface/safetensors)

---

## 💡 Executive Summary & Business Impact

This repository showcases an end-to-end **Latent Flow Matching** generative model designed to synthesize high-quality hand-written digit representations (MNIST). By partitioning the modeling problem into a two-stage pipeline—compressing raw image data into a continuous latent space and learning the data distribution via continuous-time flows—we achieve state-of-the-art generation fidelity with highly optimized computational footprints.

### 🌟 Business Value & Real-World Use Cases
While evaluated on handwriting synthesis, this exact architecture translates directly to critical industry challenges:

1. **Massive Compute & Cost Reductions:** By training our Flow Model in a highly compressed 12-dimensional latent space instead of the original 784-dimensional pixel space, we achieved a **98.47% reduction in dimensionality**. In commercial production environments, this translates directly to lower GPU inference costs, smaller model memory footprints, and sub-millisecond generation times suitable for edge-device deployment.
2. **Synthetic Data Augmentation:** In highly constrained domains (e.g., medical imaging classification, manufacturing defect detection, financial fraud analysis), obtaining labeled datasets is expensive or highly regulated. This generative pipeline can produce diverse, high-fidelity synthetic samples to robustly train downstream models, resolving severe class imbalance issues.
3. **Advanced Anomaly & Outlier Detection:** The Convolutional Variational Autoencoder (VAE) component learns the tight, structured manifold of normal data. In an industrial or cyber-security setting, high VAE reconstruction errors or unusual trajectories in the flow model can flag defective parts, fraudulent transactions, or malicious system behaviors in real-time.

---

## 🏗️ System Architecture

Our solution splits the generative process into two optimized modules, separating *representation learning* from *distribution modeling*:

```mermaid
flowchart TD
    subgraph VAE Training (Dimensionality Reduction)
        Image[Raw Images 28x28] --> Encoder[Convolutional Encoder]
        Encoder --> LatentParams[mu, log_var]
        LatentParams --> TanhConstraint[Tanh Constraint * 4.0]
        TanhConstraint --> LatentSample[Latent Vector z1 dim=12]
        LatentSample --> Decoder[Transpose Convolutional Decoder]
        Decoder --> ReconImage[Reconstructed Images 28x28]
    end
    
    subgraph Flow Matching Training (Latent Space Density Estimation)
        z0[Gaussian Noise z0 dim=12] --> Interp[Linear Interpolation xt = t*z1 + 1-t*z0]
        LatentSample --> Interp
        Interp --> FlowModel[MLP Vector Field Model]
        Time[Time Step t] --> FlowModel
        FlowModel --> PredVelocity[Predicted Velocity vt]
        TargetVelocity[Target Velocity ut = z1 - z0] --> Loss[MSE Loss]
        PredVelocity --> Loss
    end
    
    subgraph Inference & Generation (Euler Integration)
        StartNoise[z0 ~ N 0, I dim=12] --> EulerIntegrator[Forward Euler ODE Solver 100 Steps]
        TrainedFlowModel[Trained Flow Model] --> EulerIntegrator
        EulerIntegrator --> GeneratedLatent[Generated Latent z1]
        GeneratedLatent --> VDecoder[Trained Decoder]
        VDecoder --> SynthImage[High-Fidelity Synthetic Image 28x28]
    end
    
    style VAE Training fill:#f9f9f9,stroke:#333,stroke-width:1px
    style Flow Matching Training fill:#f5faff,stroke:#0052cc,stroke-width:1px
    style Inference & Generation fill:#f6fff5,stroke:#2eb82e,stroke-width:1px
```

### 1. Convolutional VAE (The Compressor)
* **Role:** Learns a structured, low-dimensional coordinate system (manifold) of the images.
* **Encoder:** Convolutional Neural Network (`Conv2d` layers, stride-based downsampling, and `SiLU` activations) that compresses $28 \times 28$ images ($784$ features) into a $12$-dimensional continuous latent space.
* **Decoder:** Transposed Convolutional Network (`ConvTranspose2d` layers) that reconstructs the original spatial dimensions from the latent vectors.

### 2. Simple Flow Model (The Navigator)
* **Role:** Learns the vector field (trajectories) inside the 12D latent space.
* **Architecture:** A Multi-Layer Perceptron (MLP) taking a noise vector and time step $t$ to compute velocity.
* **Objective:** Operates on the **Flow Matching** paradigm. Rather than training complex diffusion reverse processes, it learns straight-line paths between standard normal noise $z_0 \sim \mathcal{N}(0, \mathbf{I})$ and compressed latent vectors $z_1 \sim q(z)$.
* **Inference:** A continuous-time ODE solver (Forward Euler integration over 100 steps) smoothly pushes a noise sphere into a highly structured representation of real images, which the VAE decoder then projects back into the pixel space.

---

## 🛠️ Advanced Deep Learning Engineering & Innovations

During model development, standard architectures suffered from typical generative instability. We resolved these by implementing modern state-of-the-art training strategies:

* **KL Annealing Schedule:** To prevent *mode collapse*—where the VAE ignores the latent space regularizer and acts as a simple deterministic autoencoder—we implemented a linear warm-up scheduler for the KL weight ($\beta$ target $= 0.001$, scaling over the first 15 epochs). This allowed the model to first master reconstructions before enforcing the Gaussian prior.
* **The "KL Vanishing" Tanh Constraint:** We diagnosed a training failure where the VAE learned to bypass regularization by setting the latent variance (`log_var`) to extreme values, leading to poor and standardless latent spaces. We solved this by adding a custom, scaled `tanh` constraint inside the forward pass:
  $$\text{log\_var} = \tanh(\text{log\_var\_raw}) \times 4.0$$
  This mathematically bounded the latent variance, forcing stable, informative representations.
* **Robust Dynamic Optimization:** We integrated PyTorch's `ReduceLROnPlateau` scheduler (decreasing the learning rate by $50\%$ after 5 epochs of flat progress) to escape sharp saddle points, and added **Gradient Clipping** ($\max\text{ norm}=1.0$) to completely eliminate exploding gradients during CNN upsampling.
* **Secure Serialization:** Moving away from standard, vulnerable Python pickle files, we serialized all model weights using **Hugging Face's `safetensors` library**, ensuring fast, cross-language, and zero-copy loading of checkpoints.

---

## 📊 Experiment Tracking & Results

We utilized **Weights & Biases (wandb)** for rigorous, metric-driven development.
* **Tracking Metrics:** Every epoch logged VAE total loss, reconstruction loss, KL loss, and Flow Model matching error directly to the cloud.
* **Visual Validation:** Plotted original image samples side-by-side with VAE reconstructions inside `wandb` in real-time, validating model convergence visually.
* **Quantitative Results:** 
  * VAE Total Loss converged stably to **0.095** (Normalized BCE + regularized KL).
  * Flow matching loss steadily fell, generating sharp, distinct, and noise-free digits.

---

## 🚀 Getting Started & Execution

### Prerequisites
Ensure you have the required packages installed in your environment:
```bash
pip install torch torchvision safetensors matplotlib wandb gdown
```

### Notebook Walkthrough (`DeepLearningContest.ipynb`)
1. **Google Colab Optimization:** The notebook is fully configured for GPU acceleration (developed and tested on NVIDIA A100 tensors).
2. **Drive Integration:** Auto-mounts Google Drive to persist trained weights without local workspace constraints.
3. **Training Pipelines:** 
   * Section 5 runs the Convolutional VAE training loop (100 epochs).
   * Section 6 extracts the latent representations and trains the MLP Flow Model (100 epochs).
4. **Final Submission Class:**
   * The file defines a unified `SubmissionInterface` class that packages the two networks.
   * On initialization, it automatically fetches our optimal, pre-trained weights from Google Drive using `gdown` and runs inference synchronously.

### Testing Generation
To generate 10 fresh, synthetic digits, simply load the prepackaged interface and run:
```python
from safetensors.torch import load_file
import torch

# Instantiate the submission interface
mysub = SubmissionInterface().to(device)

# Generate novel images from latent space
generated_samples = mysub.generate_samples(n_samples=10, n_steps=100)
```

---

## 👥 Authors
* **Elijah Konkle** (elijah.konkle@belmont.edu)
* **Evan Dant** (evan.dant@belmont.edu)

*For recruitment inquiries, architectural deep-dives, or collaboration requests, please open an issue in this repository or contact the authors directly.*
