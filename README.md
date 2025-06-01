# Curriculum-Based Teacher Scheduling (CTS) for Efficient Visual Computing

This repository contains an implementation of a **Curriculum-Based Teacher Scheduling** knowledge distillation framework for visual computing tasks, leveraging Hugging Face pretrained Vision Transformer models and PyTorch.

---

## Overview

Knowledge distillation traditionally uses a static teacher-student setup. CTS improves on this by dynamically selecting specialized teacher models during training based on the visual complexity of input samples and the student's learning stage. This curriculum-based approach aims to improve student model performance and training efficiency.

---

## Features

- **Multiple specialized teachers**: Edge, texture, semantic, and multi-scale experts using pretrained ViT models.
- **Curriculum Scheduler**: Dynamically activates relevant teachers per sample and epoch.
- **Visual complexity estimation**: Uses gradient-based metrics to adaptively select teachers.
- **Combined loss**: Distillation loss weighted by teacher relevance plus standard classification loss.
- **Uses Hugging Face `transformers` and `datasets` libraries** for ease of use and reproducibility.

---

## Requirements

- Python 3.8+
- PyTorch 1.10+
- Transformers
- Datasets
- Torchvision
- NumPy

Install dependencies:

```bash
pip install torch torchvision transformers datasets numpy
