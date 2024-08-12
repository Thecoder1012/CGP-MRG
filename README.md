# **Cutting Costs, Not Corners: Multi-Modal Medical Report Generation for Alzheimer’s Disease with Fine-Tuned Large Language Models**
---
![response2](https://github.com/user-attachments/assets/45c0660c-75c6-4599-ab17-a0c079576c7e)



**CGP-MRG** (Cognition-Guided Prompting for Medical Report Generation) is a cutting-edge approach designed to generate accurate, cost-effective, and efficient medical reports for Alzheimer's Disease. Leveraging fine-tuned Large Language Models (LLMs), this project aims to deliver high-quality outputs while maintaining cost efficiency.

[![CGP-MRG_v2-1](https://github.com/user-attachments/assets/28d2dc61-9f92-42ab-a9f9-5cc42c0e7f81)](#)

## 🚀 **Installation**
To set up the environment, follow these steps:

```bash
# Clone the repository

# Navigate to the project directory

# Install the required packages
pip install -r requirements.txt
```

## 📖 **Usage**

### Run Inference on the ADMR-Test Dataset:
```bash
python3 ./LLM-Codes/infer.py
```

### Use Direct Probabilities of Cognitive Status:
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
Download the model probabilities:
```bash
./dataset/probs.txt
```

---

## **Performance Comparison**

Here is a comparison of various LLaMA models, including both pre-trained and fine-tuned versions:

| Metric          | Pretrained Llama 2 | Fine-tuned Llama 2 | Pretrained Llama 3 | Fine-tuned Llama 3 |
|-----------------|--------------------|--------------------|--------------------|--------------------|
| Perplexity      | 2.3045              | 1.9081              | 2.1532              | **1.8520**          |
| BLEU            | 0.0041              | 0.2350              | 0.1584              | **0.2646**          |
| ROUGE-1         | 0.4892              | 0.5198              | 0.5103              | **0.5279**          |
| ROUGE-2         | 0.2187              | 0.2432              | 0.2398              | **0.2477**          |
| ROUGE-L         | 0.3312              | 0.3589              | 0.3521              | **0.3656**          |
| BERTScore-P     | 0.7212              | **0.9178**          | 0.9068              | **0.9154**          |
| BERTScore-R     | 0.6939              | 0.9068              | 0.9102              | **0.9115**          |
| BERTScore-F1    | 0.7058              | 0.9120              | 0.9040              | **0.9132**          |
| Distinct-1      | 0.7144              | 0.7541              | 0.7589              | **0.7758**          |
| Distinct-2      | 0.8261              | 0.9695              | 0.9672              | **0.9683**          |
| Repetition Rate | 0.3060              | **0.2411**          | 0.2459              | **0.2242**          |
| Length Ratio    | 0.7281              | 0.7814              | 0.8726              | **0.9798**          |

**Bold** - best performance. **Underlined** - second-best performance for each metric.

---

## **PCoT-SP: Progressive-LLM-guided CoT Prompt with Structured-Section-Phrase**

[![PCoT-SP-1](https://github.com/user-attachments/assets/5ebd1e1b-6e27-4f71-92a5-e4d798a3b654)](#)

**PCoT-SP** is a key component of our methodology, enhancing the medical report generation process through structured prompting and progressive refinement.

## **Visual Performance Insights**
An overview of various performance metrics for different models, including GPU utilization, memory access time, and power usage.

![low_cost](https://github.com/user-attachments/assets/c1dcec65-4f7b-4353-8a7a-e7f2177db3ac)

A special thanks to **Claude** 🤖 and **MedGPT** 🩺 for their support and insights in this project.

---

Feel free to contribute to the project or raise issues. Your feedback is valuable!

---

**Star** ⭐ this repository if you found it helpful!
