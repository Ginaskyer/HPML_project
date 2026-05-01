# Efficient LLM Inference via Quantization and LoRA Adaptation

## 1. Project Description

Large Language Models (LLMs) have demonstrated strong performance across a wide range of natural language tasks, but their deployment is often constrained by high computational cost, memory consumption, and inference latency.  
This project investigates how to improve inference efficiency for LLMs through the combination of:

- **Quantization**
- **LoRA (Low-Rank Adaptation)**
- **QLoRA (Quantized Low-Rank Adaptation)**

Using **Qwen3-0.6B** as the experimental backbone model, we implement and compare multiple fine-tuning and deployment strategies to analyze the trade-offs between:

- model perplexity
- inference latency
- throughput
- GPU memory usage
- trainable parameter efficiency

The project also includes an interactive benchmark dashboard for visualizing and comparing all experiment results.

---

## 2. Project Milestones and Completion Status

| Milestone | Description | Status |
|-----------|-------------|--------|
| Milestone 1 | Literature review on quantization and parameter-efficient fine-tuning | Completed |
| Milestone 2 | Baseline Qwen3-0.6B environment setup and model loading | Completed |
| Milestone 3 | Full fine-tuning baseline implementation | Completed |
| Milestone 4 | LoRA adaptation pipeline development | Completed |
| Milestone 5 | 4-bit QLoRA quantized fine-tuning implementation | Completed |
| Milestone 6 | Unified inference evaluation for perplexity, latency, and throughput | Completed |
| Milestone 7 | Merged checkpoint generation and validation | Completed |
| Milestone 8 | Interactive benchmark dashboard construction | Completed |
| Milestone 9 | Final experimental analysis and report preparation | Completed |

---

## 3. Repository and Code Structure

```
Efficient-LLM-Inference/
├── baseline/                     # Full fine-tuning baseline implementation
│   ├── config.py                # Training/evaluation configuration
│   ├── model_utils.py           # Model loading and utility functions
│   ├── train.py                # Full fine-tuning training script
│   ├── evaluate.py             # Baseline evaluation script
│   └── outputs/                # Saved checkpoints and logs
│
├── qlora/                  # LoRA / QLoRA optimized implementation
│   ├── train.py
│   ├── evaluate.py
│   ├── model_utils.py
│   ├── config.py
│   └── outputs/
│
├── dashboard/                     # Web-based benchmark visualization
│
├── figures/                       # Charts, tables, and screenshots
│
├── requirements.txt
└── README.md
```

### Directory Overview

* **baseline/**
  Contains the full fine-tuning baseline used as the reference model for comparison.

* **main_version/**
  Contains LoRA and QLoRA fine-tuning implementations, quantized loading utilities, and merged checkpoint evaluation.

* **dashboard/**
  Provides an interactive web dashboard to visualize all experimental metrics.

* **figures/**
  Stores result plots, benchmark comparisons, and dashboard screenshots used in the report.

---

## 4. Example Commands to Execute the Code

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Full Fine-Tuning Baseline

```bash
cd baseline
# Train with BF16 precision (recommended)
python train.py --precision bf16 --output_dir ./outputs/bf16_run
# Train with FP32 precision
python train.py --precision fp32 --output_dir ./outputs/fp32_run

# Evaluate the original pretrained model
python evaluate.py --precision bf16
# Evaluate the saved fine-tuned BF16 checkpoint
python evaluate.py --ckpt_path ./outputs/bf16_run --precision bf16
```

### Run LoRA / QLoRA Fine-Tuning

```bash
cd main_version
python train.py
python evaluate.py
```

### Evaluate Merged Quantized Checkpoint

```bash
cd main_version
python evaluate.py --merged_path outputs/merged
```

### Launch Visualization Dashboard

```bash
cd dashboard
npm install
npm run dev
```

---

## 5. Experimental Results and Observations

### 5.1 Quantitative Comparison

| Method           | Perplexity ↓ | Latency (s) ↓ | Throughput (tokens/s) ↑ | GPU Memory (GB) ↓ | Trainable Params |
| ---------------- | ------------ | ------------- | ----------------------- | ----------------- | ---------------- |
| Full Fine-Tuning | XX.XX        | XX.XX         | XX.XX                   | XX.XX             | 596M             |
| LoRA             | XX.XX        | XX.XX         | XX.XX                   | XX.XX             | XXM              |
| QLoRA            | XX.XX        | XX.XX         | XX.XX                   | XX.XX             | XXM              |

---

### 5.2 Experimental Charts

#### Perplexity Comparison

![Perplexity Comparison](figures/perplexity.png)

#### Inference Throughput Comparison

![Throughput Comparison](figures/throughput.png)

#### GPU Memory Consumption

![GPU Memory](figures/gpu_memory.png)

---

### 5.3 Key Observations

* Quantization significantly reduces GPU memory consumption and makes large model deployment feasible on limited hardware.
* LoRA greatly decreases the number of trainable parameters while preserving competitive language modeling performance.
* QLoRA achieves the best trade-off between memory efficiency and model quality among all tested approaches.
* Merged checkpoints allow efficient standalone inference without requiring external LoRA adapters.
* The benchmark dashboard provides an intuitive visualization for comparing all experimental settings.

---

## 6. Authors

* Victoria Logan
* [Team Member Name]

Course Project for High Performance Machine Learning

---

## 7. GitHub Repository

Source code, experiment scripts, and visualization dashboard are all maintained in this repository.

```
```