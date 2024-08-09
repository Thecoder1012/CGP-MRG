import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Function
from torch.utils.data import DataLoader, random_split, TensorDataset
from torch.utils.data.dataloader import default_collate
import os
import pandas as pd
import torch.nn.functional as F
import csv
import torch.nn as nn
import re
from metrics import *
import warnings
warnings.simplefilter('ignore')

from unsloth import FastLanguageModel
def pad_sequence(sequences):
    max_size = max([s.size() for s in sequences])
    padded_sequences = [F.pad(s, (0, max_size[2] - s.size(2), 0, max_size[1] - s.size(1), 0, max_size[0] - s.size(0))) for s in sequences]
    return torch.stack(padded_sequences)

def my_collate_fn(batch):
    batch = {k: [d[k] for d in batch] for k in batch[0]}
    for k in batch:
        if k == 'image_data':
            batch[k] = pad_sequence(batch[k])
        else:
            batch[k] = default_collate(batch[k])
    return batch

max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = torch.float16 # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

model_llm, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-2-7b",
    max_seq_length = max_seq_length,
    dtype = dtype,
    cache_dir = "/LLM/llama2_7b_experiment",
    load_in_4bit = load_in_4bit,
)

EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
def formatting_prompts_func(examples):
    instructions = examples["Instruction"]
    inputs       = examples["Input"]
    outputs      = examples["Output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }

def extract_response(text):
    parts = text.split("###")
    response_part = parts[-1]
    response = response_part[12:]
    response = response[:-17]
    return response

def trim_to_last_sentence(text):
    # List of sentence-ending punctuation
    end_punctuation = ['.', '!', '?']

    # Remove trailing whitespace
    text = text.rstrip()

    # Find the last occurrence of each sentence-ending punctuation
    last_positions = [text.rfind(punct) for punct in end_punctuation]

    # Get the position of the last sentence-ending punctuation
    last_sentence_end = max(last_positions)

    # If we found a sentence-ending punctuation, trim the string
    if last_sentence_end != -1:
        return text[:last_sentence_end + 1]
    else:
        return text

# Extract probabilities from the input string
def parse_probabilities(probability_string):
    prob_dict = {}
    lines = probability_string.split('\n')
    for line in lines:
        # print(line)
        class_name, prob = line.split(': ')
        prob_dict[class_name] = float(prob.strip('%'))
    return prob_dict

# Determine the best class and its probability
def get_best_class(prob_dict):
    best_class = max(prob_dict, key=prob_dict.get)
    best_prob = prob_dict[best_class]
    return best_class, best_prob

# Find top 10 matches within a range of 0.5% for the best class
def find_top_matches(csv_inputs, csv_outputs, best_class, best_prob, tolerance=0.5, top_k=10):
    matches = []
    for index, prob_dict in enumerate(csv_inputs):
        # print(prob_dict)
        if best_class == 'Dementia':
            best_dict_class = "Alzheimer's Disease"
        elif best_class == "CN":
            best_dict_class = "Cognitive Normal"
        else:
            best_dict_class = "Mild Cognitive Impairment"

        if abs(prob_dict[best_dict_class] - best_prob) <= tolerance:
            matches.append((index, prob_dict, csv_outputs[index]))
    matches_sorted = sorted(matches, key=lambda x: abs(x[1][best_dict_class] - best_prob))
    top_matches = matches_sorted[:top_k]
    return top_matches

FastLanguageModel.for_inference(model_llm)

best_perplexities = []
best_bleu_scores = []
best_rouge_scores = []
best_bertscore_p = []
best_bertscore_r = []
best_bertscore_f1 = []
best_distinct1_scores = []
best_distinct2_scores = []
# best_moverscore_scores = []
best_repetition_rates = []
best_length_ratios = []

with open("./probs.txt", 'r') as file:
    contents = file.read()

contents_list = contents.strip().split('\n\n')

csv_file_path = './test_detailed_medical_reports_dataset19_07_2024.csv'
csv_data = pd.read_csv(csv_file_path)
csv_inputs = csv_data['Input'].apply(parse_probabilities)
csv_outputs = csv_data['Output']

cnt = 1
for formatted_output in contents_list:
    print(formatted_output)

    # print("preds:", formatted_output)

    # csv_file_path = './LLM/baseline_comp/report.csv'
    output_file_path = f"./pretrained/llama2/{cnt}.txt"
    cnt = cnt + 1

    input_llm = formatted_output

    # print(f"Classification Result: Class {pred.item()}")

    torch.cuda.empty_cache()

    inputs = tokenizer(
    [
        alpaca_prompt.format(
            "Make a medical report with summary.",
            input_llm,
            "",
        )
    ], return_tensors="pt").to("cuda")

    outputs_llm = model_llm.generate(**inputs, max_new_tokens=2048, use_cache=True)

    response = extract_response(str(tokenizer.batch_decode(outputs_llm)))

    response = trim_to_last_sentence(response)
    print(response)



    # Parse the input string
    input_probs = parse_probabilities(input_llm)
    best_class, best_prob = get_best_class(input_probs)



    # Get the top 10 matches
    top_matches = find_top_matches(csv_inputs, csv_outputs, best_class, best_prob)
    if len(top_matches) != 0:


        # Initialize lists to store scores and outputs for each metric
        perplexities = []
        bleu_scores = []
        rouge_scores = []
        bertscore_p = []
        bertscore_r = []
        bertscore_f1 = []
        distinct1_scores = []
        distinct2_scores = []
        # moverscore_scores = []
        repetition_rates = []
        length_ratios = []

        # Initialize lists to store corresponding outputs
        outputs_for_perplexities = []
        outputs_for_bleu = []
        outputs_for_rouge = []
        outputs_for_bertscore_p = []
        outputs_for_bertscore_r = []
        outputs_for_bertscore_f1 = []
        outputs_for_distinct1 = []
        outputs_for_distinct2 = []
        # outputs_for_moverscore = []
        outputs_for_repetition = []
        outputs_for_length_ratios = []

        # Loop through the top matches
        for index, prob_dict, output in top_matches:
            # Calculate scores
            perplexity = calculate_perplexity(model_llm, tokenizer, response)
            bleu = calculate_bleu(output, response)
            rouge = calculate_rouge(output, response)
            # print(rouge['rouge1'])
            P, R, F1 = calculate_bertscore([output], [response])
            distinct1 = calculate_distinct_n(response.split(), 1)
            distinct2 = calculate_distinct_n(response.split(), 2)
            # moverscore = calculate_moverscore([output], [response])
            repetition_rate = calculate_repetition_rate(response)
            length_ratio = calculate_length_ratio(output, response)

            # Append scores to their lists
            perplexities.append(perplexity)
            bleu_scores.append(bleu)
            rouge_scores.append(rouge)
            bertscore_p.append(P)
            bertscore_r.append(R)
            bertscore_f1.append(F1)
            distinct1_scores.append(distinct1)
            distinct2_scores.append(distinct2)
            # moverscore_scores.append(moverscore)
            repetition_rates.append(repetition_rate)
            length_ratios.append(length_ratio)

            # Append outputs to their lists
            outputs_for_perplexities.append(output)
            outputs_for_bleu.append(output)
            outputs_for_rouge.append(output)
            outputs_for_bertscore_p.append(output)
            outputs_for_bertscore_r.append(output)
            outputs_for_bertscore_f1.append(output)
            outputs_for_distinct1.append(output)
            outputs_for_distinct2.append(output)
            # outputs_for_moverscore.append(output)
            outputs_for_repetition.append(output)
            outputs_for_length_ratios.append(output)

        # Find the best score and corresponding output for each metric
        best_perplexity_index = perplexities.index(min(perplexities))
        best_bleu_index = bleu_scores.index(max(bleu_scores))
        # best_rouge_index = rouge_scores.index(max(rouge_scores['']))
        best_bertscore_p_index = bertscore_p.index(max(bertscore_p))
        best_bertscore_r_index = bertscore_r.index(max(bertscore_r))
        best_bertscore_f1_index = bertscore_f1.index(max(bertscore_f1))
        best_distinct1_index = distinct1_scores.index(max(distinct1_scores))
        best_distinct2_index = distinct2_scores.index(max(distinct2_scores))
        # best_moverscore_index = moverscore_scores.index(max(moverscore_scores))
        best_repetition_rate_index = repetition_rates.index(min(repetition_rates))
        best_length_ratio_index = length_ratios.index(min(length_ratios))

        best_perplexities.append(min(perplexities))
        best_bleu_scores.append(max(bleu_scores))
        # best_rouge.append(max(rouge_scores))
        best_bertscore_p.append(max(bertscore_p))
        best_bertscore_r.append(max(bertscore_r))
        best_bertscore_f1.append(max(bertscore_f1))
        best_distinct1_scores.append(max(distinct1_scores))
        best_distinct2_scores.append(max(distinct2_scores))
        # best_moverscore_scores.append(max(moverscore_scores))
        best_repetition_rates.append(min(repetition_rates))
        best_length_ratios.append(min(length_ratios))


        # Create dictionaries to store best scores and their corresponding outputs
        best_metrics = {
            "perplexity": {
                "score": perplexities[best_perplexity_index],
                "output": outputs_for_perplexities[best_perplexity_index]
            },
            "bleu": {
                "score": bleu_scores[best_bleu_index],
                "output": outputs_for_bleu[best_bleu_index]
            },
            # "rouge": {
            #     "score": rouge_scores[best_rouge_index],
            #     "output": outputs_for_rouge[best_rouge_index]
            # },
            "bertscore_p": {
                "score": bertscore_p[best_bertscore_p_index],
                "output": outputs_for_bertscore_p[best_bertscore_p_index]
            },
            "bertscore_r": {
                "score": bertscore_r[best_bertscore_r_index],
                "output": outputs_for_bertscore_r[best_bertscore_r_index]
            },
            "bertscore_f1": {
                "score": bertscore_f1[best_bertscore_f1_index],
                "output": outputs_for_bertscore_f1[best_bertscore_f1_index]
            },
            "distinct1": {
                "score": distinct1_scores[best_distinct1_index],
                "output": outputs_for_distinct1[best_distinct1_index]
            },
            "distinct2": {
                "score": distinct2_scores[best_distinct2_index],
                "output": outputs_for_distinct2[best_distinct2_index]
            },
            # "moverscore": {
            #     "score": moverscore_scores[best_moverscore_index],
            #     "output": outputs_for_moverscore[best_moverscore_index]
            # },
            "repetition_rate": {
                "score": repetition_rates[best_repetition_rate_index],
                "output": outputs_for_repetition[best_repetition_rate_index]
            },
            "length_ratio": {
                "score": length_ratios[best_length_ratio_index],
                "output": outputs_for_length_ratios[best_length_ratio_index]
            }
        }

        # Open the file and write the best metrics
        with open(output_file_path, 'w') as file:
            # file.write(f"Classification Result: Class {pred.item()}\n")
            file.write(f"Model Response:\n{response}\n")
            file.write("\nBest Metrics:\n")
            for metric, data in best_metrics.items():
                file.write(f"\n{metric}:\n")
                file.write(f"  Score: {data['score']}\n")
                file.write(f"  Output: {data['output']}\n")


# Define the output file path for the averages
averages_output_file = "./pretrained/llama2/average_best_scores_llama2.txt"

# Open the file and write the averages
with open(averages_output_file, 'w') as file:
    file.write("Average Best Scores:\n")
    file.write("scores ==================\n")
    file.write(f"Perplexity: {sum(best_perplexities) / len(best_perplexities)}\n")
    file.write(f"BLEU: {sum(best_bleu_scores) / len(best_bleu_scores)}\n")
    # file.write(f"ROUGE: {sum(all_best_rouge_scores) / len(all_best_rouge_scores)}\n") # Uncomment if needed
    file.write(f"BERTScore Precision: {sum(best_bertscore_p) / len(best_bertscore_p)}\n")
    file.write(f"BERTScore Recall: {sum(best_bertscore_r) / len(best_bertscore_r)}\n")
    file.write(f"BERTScore F1: {sum(best_bertscore_f1) / len(best_bertscore_f1)}\n")
    file.write(f"Distinct-1: {sum(best_distinct1_scores) / len(best_distinct1_scores)}\n")
    file.write(f"Distinct-2: {sum(best_distinct2_scores) / len(best_distinct2_scores)}\n")
    # file.write(f"MoverScore: {sum(best_moverscore_scores) / len(best_moverscore_scores)}\n")
    file.write(f"Repetition Rate: {sum(best_repetition_rates) / len(best_repetition_rates)}\n")
    file.write(f"Length Ratio: {sum(best_length_ratios) / len(best_length_ratios)}\n")
