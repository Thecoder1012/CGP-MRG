# CGP-MRG: Cognition-Guided Prompting for Alzheimer's Disease Medical Report Generation

## Cutting Costs, Not Corners: Multi-Modal Medical Report Generation for Alzheimer's Disease with Fine-Tuned Large Language Models

[CGP-MRG Architecture]

CGP-MRG (Cognition-Guided Prompting for Medical Report Generation) is an innovative approach to generating medical reports for Alzheimer's Disease using fine-tuned Large Language Models. This project aims to provide accurate, cost-effective, and efficient medical report generation while maintaining high-quality outputs.

### Installation

$ clone the git
$ cd CGP-MRG
$ pip install -r requirements.txt

### Usage

To run the code on the ADMR-Test dataset:
$ python ./LLM-Codes/infer.py

To use direct probabilities from classification models:
$ python ./LLM-Codes/baseline_comp_sample.py

### Models and Data

- LoRA models: https://drive.google.com/drive/folders/1JjG6C0xO5INWj_MtnqE-76KCp1F510g1?usp=sharing
- Interactive notebook in Google Colab for pretrained baselines (Llama 3 8b, Llama 2 7b, Mistral 7b, Gemma 7b, Phi3 Medium)
- ADMR-Test dataset: ./dataset/test.csv
- Classification model probabilities: ./dataset/probs.txt

### PCoT-SP: Progressive-LLM-guided CoT prompt with structured-Section-Phrase

[PCoT-SP Architecture]

PCoT-SP is an integral part of our methodology, enhancing the report generation process through structured prompting and progressive refinement.

---
