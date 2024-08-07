import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, BartForConditionalGeneration, BartTokenizer
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from unsloth import FastLanguageModel
import os
import torch
import bert_score
from collections import Counter
from tqdm import tqdm
# from bleurt import score as bleurt_score
from moverscore import get_idf_dict, word_mover_score
from typing import List
import numpy as np

# Ensure only GPU 0 is visible to the script
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Load the CSV file
csv_path = '/DATA1/arkaprabha/LLM/dataset/test_detailed_medical_reports_dataset19_07_2024.csv'
df = pd.read_csv(csv_path)
output_file_path = '../scores/output_scores_llama3_19_07_2024.txt'
# Load the model and tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="/DATA1/arkaprabha/LLM/v3_models/lora_model_llama3_15_07_2024",
    max_seq_length=4000,
    dtype=torch.float16,
    cache_dir="/DATA1/arkaprabha/LLM/llama3_8b_experiment_2",
    load_in_4bit=True,
)
model = model.to("cuda:0")

# Load BLEURT
# bleurt_checkpoint = "bleurt-base-128"  # You may need to download this checkpoint
# bleurt_scorer = bleurt_score.BleurtScorer(bleurt_checkpoint)

# Load BARTScore
bart_scorer = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn', cache_dir = "/DATA1/arkaprabha/LLM/BART/").to("cuda:0")
bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn', cache_dir = "/DATA1/arkaprabha/LLM/BART/")

# Function to extract the response
def extract_response(text):
    parts = text.split("###")
    response_part = parts[-1]
    response = response_part[12:]
    response = response[:-17]
    return response

def trim_to_last_sentence(text):
    end_punctuation = ['.', '!', '?']
    text = text.rstrip()
    last_positions = [text.rfind(punct) for punct in end_punctuation]
    last_sentence_end = max(last_positions)
    if last_sentence_end != -1:
        return text[:last_sentence_end + 1]
    else:
        return text

# Function to calculate perplexity
def calculate_perplexity(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs.input_ids.to(model.device)
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    loss = outputs.loss
    perplexity = torch.exp(loss)
    return perplexity.item()

# Function to calculate BLEU
def calculate_bleu(reference, hypothesis):
    return sentence_bleu([reference.split()], hypothesis.split())

# Function to calculate ROUGE
def calculate_rouge(reference, hypothesis):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    return scores

# Function to calculate BERTScore
def calculate_bertscore(references, hypotheses):
    P, R, F1 = bert_score.score(hypotheses, references, lang='en', device='cuda:0')
    return P.mean().item(), R.mean().item(), F1.mean().item()

# Function to calculate distinct-n
def calculate_distinct_n(hypotheses, n):
    n_grams = [tuple(hypotheses[i:i+n]) for i in range(len(hypotheses)-n+1)]
    return len(set(n_grams)) / len(n_grams) if n_grams else 0

# Function to calculate BLEURT
def calculate_bleurt(references: List[str], candidates: List[str]) -> float:
    scores = bleurt_scorer.score(references=references, candidates=candidates)
    return sum(scores) / len(scores)

# Function to calculate MoverScore
def calculate_moverscore(references: List[str], hypotheses: List[str]) -> float:
    idf_dict_ref = get_idf_dict(references)
    idf_dict_hyp = get_idf_dict(hypotheses)
    scores = word_mover_score(references, hypotheses, idf_dict_ref, idf_dict_hyp, stop_words=[], n_gram=1, remove_subwords=True)
    return sum(scores) / len(scores)

# Function to calculate BARTScore
def calculate_bartscore(reference: str, hypothesis: str) -> float:
    with torch.no_grad():
        inputs = bart_tokenizer(reference, return_tensors='pt').to("cuda:0")
        outputs = bart_scorer.generate(**inputs)
        score = bart_scorer(inputs.input_ids, outputs).loss.item()
    return -score  # Negative log likelihood

# Function to calculate Repetition Rate
def calculate_repetition_rate(text: str) -> float:
    words = text.split()
    if not words:
        return 0
    return (len(words) - len(set(words))) / len(words)

# Function to calculate Length Ratio
def calculate_length_ratio(reference: str, hypothesis: str) -> float:
    return len(hypothesis.split()) / len(reference.split())

# Alpaca prompt template
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

# Initialize score lists
perplexities = []
bleu_scores = []
rouge_scores = []
bertscore_p = []
bertscore_r = []
bertscore_f1 = []
distinct1_scores = []
distinct2_scores = []
# bleurt_scores = []
moverscore_scores = []
# bartscore_scores = []
repetition_rates = []
length_ratios = []

# Iterate over each row in the CSV
for index, row in tqdm(df.iterrows(), total=df.shape[0]):
    instruction = row['Instruction']
    input_text = row['Input']
    target_output = row['Output']

    # Prepare the input for the model
    prompt = alpaca_prompt.format(instruction, input_text, "")
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda:0")

    # Generate the model output
    outputs = model.generate(**inputs, max_new_tokens=1000, use_cache=True)
    generated_output = extract_response(str(tokenizer.batch_decode(outputs)))
    generated_output = trim_to_last_sentence(generated_output)

    print("=============================Output=================")
    print(generated_output)
    print("=============================Target=================")
    print(target_output)
    print("====================================================")

    # Calculate scores
    perplexities.append(calculate_perplexity(model, tokenizer, generated_output))
    bleu_scores.append(calculate_bleu(target_output, generated_output))
    rouge_scores.append(calculate_rouge(target_output, generated_output))
    P, R, F1 = calculate_bertscore([target_output], [generated_output])
    bertscore_p.append(P)
    bertscore_r.append(R)
    bertscore_f1.append(F1)
    distinct1_scores.append(calculate_distinct_n(generated_output.split(), 1))
    distinct2_scores.append(calculate_distinct_n(generated_output.split(), 2))
    # bleurt_scores.append(calculate_bleurt([target_output], [generated_output]))
    moverscore_scores.append(calculate_moverscore([target_output], [generated_output]))
    # bartscore_scores.append(calculate_bartscore(target_output, generated_output))
    repetition_rates.append(calculate_repetition_rate(generated_output))
    length_ratios.append(calculate_length_ratio(target_output, generated_output))

    torch.cuda.empty_cache()

# Calculate average scores
avg_perplexity = sum(perplexities) / len(perplexities)
avg_bleu = sum(bleu_scores) / len(bleu_scores)
avg_rouge_1 = sum([score['rouge1'].fmeasure for score in rouge_scores]) / len(rouge_scores)
avg_rouge_2 = sum([score['rouge2'].fmeasure for score in rouge_scores]) / len(rouge_scores)
avg_rouge_l = sum([score['rougeL'].fmeasure for score in rouge_scores]) / len(rouge_scores)
avg_bertscore_p = sum(bertscore_p) / len(bertscore_p)
avg_bertscore_r = sum(bertscore_r) / len(bertscore_r)
avg_bertscore_f1 = sum(bertscore_f1) / len(bertscore_f1)
avg_distinct1 = sum(distinct1_scores) / len(distinct1_scores)
avg_distinct2 = sum(distinct2_scores) / len(distinct2_scores)
# avg_bleurt = sum(bleurt_scores) / len(bleurt_scores)
avg_moverscore = sum(moverscore_scores) / len(moverscore_scores)
# avg_bartscore = sum(bartscore_scores) / len(bartscore_scores)
avg_repetition_rate = sum(repetition_rates) / len(repetition_rates)
avg_length_ratio = sum(length_ratios) / len(length_ratios)

# Define the output file path


# Write the average scores to the file
with open(output_file_path, 'w') as file:
    file.write(f"Average Perplexity: {avg_perplexity}\n")
    file.write(f"Average BLEU: {avg_bleu}\n")
    file.write(f"Average ROUGE-1: {avg_rouge_1}\n")
    file.write(f"Average ROUGE-2: {avg_rouge_2}\n")
    file.write(f"Average ROUGE-L: {avg_rouge_l}\n")
    file.write(f"Average BERTScore P: {avg_bertscore_p}\n")
    file.write(f"Average BERTScore R: {avg_bertscore_r}\n")
    file.write(f"Average BERTScore F1: {avg_bertscore_f1}\n")
    file.write(f"Average Distinct-1: {avg_distinct1}\n")
    file.write(f"Average Distinct-2: {avg_distinct2}\n")
    # file.write(f"Average BLEURT: {avg_bleurt}\n")
    file.write(f"Average MoverScore: {avg_moverscore}\n")
    # file.write(f"Average BARTScore: {avg_bartscore}\n")
    file.write(f"Average Repetition Rate: {avg_repetition_rate}\n")
    file.write(f"Average Length Ratio: {avg_length_ratio}\n")

print("Evaluation complete. Results written to", output_file_path)