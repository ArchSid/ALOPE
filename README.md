# ALOPE

ALOPE is an adaptive layer-optimization framework that enhances quality estimation (QE) for machine translation using large language models (LLMs). It restructures Transformer representations through layer-wise adaptation and integrates low-rank adapters (LoRA) with regression heads, enabling improved regression-based prediction, especially for low-resource languages. ALOPE also introduces dynamic weighting and multi-head regression strategies, adaptively combining information from multiple Transformer layers. The framework is designed to be easily integrated into existing LLMs, enabling robust reference-less quality estimation.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/ArchSid/ALOPE.git
cd ALOPE
```

### 2.Create a new Conda virtual environment
```bash
conda create -n myenv python=3.10
conda activate myenv
pip install -r requirements.txt
```


