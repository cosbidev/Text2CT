# Text-to-CT Generation via 3D Latent Diffusion Model

This repository hosts the official project page for our work:

**[Text-to-CT Generation via 3D Latent Diffusion Model with Contrastive Vision-Language Pretraining](https://arxiv.org/abs/2506.00633)**  
Molino, D., Caruso, C. M., Ruffini, F., Soda, P., Guarrasi, V. (2025)

---

## 🧠 Model Overview
Our approach combines:
- A **3D CLIP-style encoder** for vision-language alignment between CT volumes and radiology reports.
- A **volumetric VAE** for latent compression of 3D CT data.
- A **latent diffusion model** with cross-attention conditioning for controllable text-to-CT generation.  

This design enables direct synthesis of anatomically consistent, semantically faithful, and high-resolution CT volumes from textual descriptions.

---

## 📦 Synthetic Dataset
We release **1,000 synthetic chest CT scans** generated with our model for the [VLM3D Challenge](https://vlm3dchallenge.com).  
➡️ Available on Hugging Face: [Synthetic Text-to-CT Dataset](https://huggingface.co/datasets/dmolino/CT-RATE_Generated_Scans)  

---

## 📜 Paper
- Preprint: [arXiv:2506.00633](https://arxiv.org/abs/2506.00633)

---

## 🚧 Code Release
The **full training and inference code** will be made available soon.  
Stay tuned for updates! ✨

---

## 📬 Contact
For questions or collaborations, please reach out to:  
**Daniele Molino** – [daniele.molino@unicampus.it](mailto:daniele.molino@unicampus.it)  

---
