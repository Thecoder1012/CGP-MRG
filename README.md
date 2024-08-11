# **Cutting Costs, Not Corners: Multi-Modal Medical Report Generation for Alzheimer’s Disease with Fine-Tuned Large Language Models**
---

**CGP-MRG** (Cognition-Guided Prompting for Medical Report Generation) is a cutting-edge approach designed to generate accurate, cost-effective, and efficient medical reports for Alzheimer's Disease. Leveraging fine-tuned Large Language Models (LLMs), this project aims to deliver high-quality outputs while maintaining cost efficiency.

[![CGP-MRG_v2-1](https://github.com/user-attachments/assets/28d2dc61-9f92-42ab-a9f9-5cc42c0e7f81)](#)

## 🚀 **Installation**
To set up the environment, follow these steps:

```bash
# Clone the repository
git clone https://github.com/your-repo/cgp-mrg.git

# Navigate to the project directory
cd cgp-mrg

# Install the required packages
pip install -r requirements.txt
```

## 📖 **Usage**

### Run Inference on the ADMR-Test Dataset:
```bash
python3 ./LLM-Codes/infer.py
```

### Use Direct Probabilities from Classification Models:
```bash
python3 ./LLM-Codes/baseline_comp_sample.py
```

### Fine-Tune LLM Models:
To fine-tune the models in our setting, use the following command:
```bash
python3 ./LLM-Codes/llama3.py
```
> **Note:** This script is configured for **LLaMA 3**. You can explore other models like **Unsloth** from _Hugging Face_ for LoRA fine-tuning.

## 📊 **Models and Data**

### LoRA Models:
Access the LoRA models through the following link:
[Google Drive - LoRA Models](https://drive.google.com/drive/folders/1JjG6C0xO5INWj_MtnqE-76KCp1F510g1?usp=sharing)

### Interactive Notebook for Pretrained Baselines:
Explore pretrained baselines like **LLaMA 3 8b, LLaMA 2 7b, Mistral 7b, Gemma 7b,** and **Phi3 Medium** using this Colab notebook:
```bash
./LLM-Codes/baseline_pretrained.ipynb
```

### ADMR-Test Dataset:
Access the ADMR-Test dataset:
```bash
./dataset/test.csv
```

### Classification Model Probabilities:
Download the classification model probabilities:
```bash
./dataset/probs.txt
```

---

## **PCoT-SP: Progressive-LLM-guided CoT Prompt with Structured-Section-Phrase**

[![PCoT-SP-1](https://github.com/user-attachments/assets/5ebd1e1b-6e27-4f71-92a5-e4d798a3b654)](#)

**PCoT-SP** is a key component of our methodology, enhancing the medical report generation process through structured prompting and progressive refinement.

---

Feel free to contribute to the project or raise issues. Your feedback is valuable!

---

**Star** ⭐ this repository if you found it helpful!

---
