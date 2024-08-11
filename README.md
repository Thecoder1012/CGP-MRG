# Cutting Costs, Not Corners: Multi-Modal Medical Report Generation for Alzheimer’s Disease with Fine-Tuned Large Language Models

## CGP-MRG: Cognition-Guided Prompting for Alzheimer's Disease Medical Report Generation

![CGP-MRG_v2-1](https://github.com/user-attachments/assets/28d2dc61-9f92-42ab-a9f9-5cc42c0e7f81)

CGP-MRG (Cognition-Guided Prompting for Medical Report Generation) is an innovative approach to generating medical reports for Alzheimer's Disease using fine-tuned Large Language Models. This project aims to provide accurate, cost-effective, and efficient medical report generation while maintaining high-quality outputs.

### Installation
````
clone the git

pip install -r requirements.txt
````
### Usage

To run the code on the ADMR-Test dataset:
```
python3 ./LLM-Codes/infer.py
````

To use direct probabilities from classification models:
````
python3 ./LLM-Codes/baseline_comp_sample.py
````

### Models and Data
- LoRA models: https://drive.google.com/drive/folders/1JjG6C0xO5INWj_MtnqE-76KCp1F510g1?usp=sharing
- Interactive notebook in Google Colab for pretrained baselines (Llama 3 8b, Llama 2 7b, Mistral 7b, Gemma 7b, Phi3 Medium)
  ````
  ./LLM-Codes/baseline_pretrained.ipynb
  ````
- ADMR-Test dataset:
  ````
  ./dataset/test.csv
  ````
- Classification model probabilities:
  ````
  ./dataset/probs.txt
  ````
### PCoT-SP: Progressive-LLM-guided CoT prompt with structured-Section-Phrase

![PCoT-SP-1](https://github.com/user-attachments/assets/5ebd1e1b-6e27-4f71-92a5-e4d798a3b654)

PCoT-SP is an integral part of our methodology, enhancing the report generation process through structured prompting and progressive refinement.

---
