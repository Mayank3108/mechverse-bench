# MechVerse-Bench

A visual multiple-choice benchmark evaluating frontier VLMs on
3D mechanical assembly spatial reasoning.

## Overview

MechVerse-Bench tests whether frontier models can reason about
mechanical assemblies from CAD-rendered images. Each question
presents a color-coded 3D assembly with a coordinate axes indicator
and asks the model to answer a spatial reasoning question by
selecting from four choices (A, B, C, D).

## Dataset

- **65 questions** across **34 unique images**
- **4 reasoning dimensions**, **13 categories**
- Random chance baseline: **25%**

### Dimensions and Categories

| Dimension | Categories |
|---|---|
| Path / Disassembly Reasoning | Removal Direction, Obstruction Identification, Multi-Direction Feasibility |
| Sequence / Precedence Reasoning | Blocking Part, First Removal, Dependency Chain Length, Independent Subassembly |
| Motion / Kinematic Reasoning | Motion Type, Primary Motion Axis, Motion Coupling |
| Joint / Constraint Reasoning | Joint Type, Joint Axis, Joint Location |

## Results (pass@2)

| Model | query_wo | query_cot |
|---|---|---|
| GPT-5.4 | 49.2% | 55.4% |
| Claude Opus 4.6 | 46.2% | 66.2% |
| Gemini 3.1 Pro | 70.8% | 70.8% |
| Random baseline | 25.0% | 25.0% |

Hardest categories: Joint Axis Identification (20-40%),
Motion Type Identification (20-60%), Obstruction Identification (20-40%)

## Project Structure

```
mechverse-bench/
├── dataset/
│   ├── annotations.json     ← 65 curated questions
│   └── images/              ← 34 CAD assembly images
├── models/
│   └── openrouter_model.py  ← unified OpenRouter wrapper
├── scripts/
│   ├── run_assessment.py    ← queries all 3 models (pass@2)
│   └── compute_scores.py    ← computes pass@2 scores
├── results/
│   └── assessment/          ← 6 result files + score report
├── oracle/
│   └── oracle_trajectory.json ← human expert solutions
└── harbor-task/             ← Harbor format task definition
    ├── instruction.md
    ├── task.toml
    ├── environment/Dockerfile
    ├── solution/solve.sh
    └── tests/test.sh + verify.py
```

## Setup

```bash
conda create -n mechverse-bench python=3.11
conda activate mechverse-bench
pip install openai pillow tqdm python-dotenv
```

Add your OpenRouter API key to `.env`:
```
OPENROUTER_API_KEY=sk-or-your-key-here
```

## Running the Benchmark

```bash
# Run all 3 models with pass@2
python scripts/run_assessment.py

# Compute pass@2 scores
python scripts/compute_scores.py
```


