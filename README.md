# Industrial Defect Detection: OpenCV + LLM Pipeline

> A closed-loop AI quality inspection system combining traditional computer vision with Large Language Models for intelligent defect detection and disposition in manufacturing environments.

## Overview

This project implements a two-stage industrial quality inspection pipeline:

1. **Few-Shot Defect Detection** — Using pretrained ResNet50/ResNet18 with prototype networks, the system classifies product images as OK/NG with as few as 3-5 reference samples per category.

2. **LLM-Powered Diagnosis & Disposition** — Detected defects are analyzed by a multimodal LLM (Qwen-VL-Max) via structured JSON Schema output, providing:
   - Defect type classification with confidence scores
   - Visual feature description
   - Production impact assessment
   - Actionable disposition recommendations

### Key Technical Highlights

- **Self-Correction via Confidence Threshold**: When any defect's confidence score falls below 0.8, the system triggers an automatic re-review with stricter prompting to reduce hallucination in edge cases.
- **Prompt Constraint Engineering**: Domain-specific prompt templates ensure the LLM generates manufacturing-relevant, actionable suggestions rather than generic responses.
- **Tool-Chain Injection**: Integration of defect reference databases and historical repair records into the LLM context, achieving 100% valid recommendation rate.

## Project Structure

```
├── quick_start_industrial.py  # Few-shot detector (ResNet50 + Prototype Network)
├── qwen.py                    # LLM inspection pipeline (Qwen-VL-Max)
├── report.html                # Sample inspection report (sanitized)
├── data/                      # Place your reference images here
│   ├── OK/                    #   Normal samples (3-10 images)
│   ├── NG_scratch/            #   Defect type 1
│   └── NG_foreign_object/     #   Defect type 2
├── .env.example               # API key configuration template
├── .gitignore
└── README.md
```

## Quick Start

### 1. Few-Shot Detection

```python
from quick_start_industrial import IndustrialFewShotDetector

detector = IndustrialFewShotDetector(backbone="resnet50")

# Register reference samples (only 3-5 needed per class!)
detector.register_class("OK", ["ok_1.jpg", "ok_2.jpg", "ok_3.jpg"])
detector.register_class("NG_scratch", ["ng_1.jpg", "ng_2.jpg"])

# Detect
result = detector.detect("test_image.jpg")
print(f"Result: {result['class']} (Confidence: {result['confidence']:.1%})")
```

### 2. LLM Diagnosis Pipeline

```bash
# Set up API key
cp .env.example .env
# Edit .env with your API key

# Run inspection
python qwen.py --image test_image.jpg
```

### 3. Run Demo

```bash
python quick_start_industrial.py
```

This generates synthetic data and demonstrates the few-shot detection workflow.

## System Architecture

```
┌──────────────┐    ┌───────────────────┐    ┌──────────────────┐
│  GigE Camera │───▶│  OpenCV Preprocess│───▶│  Few-Shot Detect  │
│  (Capture)   │    │  (ROI, Enhance)   │    │  (ResNet50+Proto) │
└──────────────┘    └───────────────────┘    └────────┬─────────┘
                                                      │
                                              OK ◄────┤────► NG
                                                      │
                                             ┌────────▼─────────┐
                                             │  Qwen-VL-Max LLM │
                                             │  (JSON Schema)    │
                                             └────────┬─────────┘
                                                      │
                                         ┌────────────▼────────────┐
                                         │  Confidence < 0.8?      │
                                         │  YES → Re-review        │
                                         │  NO  → Generate Report  │
                                         └─────────────────────────┘
```

## Requirements

```
torch>=1.12
torchvision>=0.13
Pillow
numpy
openai  # for Qwen API calls
python-dotenv
```

## Notes

- Factory images are excluded from this repository due to NDA constraints.
- The demo uses synthetic data to showcase the detection workflow.
- For real deployment, integrate with PyQt5/PySide6 GUI (see code comments for example).

## License

MIT
