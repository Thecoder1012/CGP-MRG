import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, default_collate
from model_v2 import MultimodalNetwork
from dataset import MultimodalDataset, PreprocessTransform
from metrics import *
from unsloth import FastLanguageModel
import warnings

# Suppress warnings for cleaner output
warnings.simplefilter('ignore')

# Set device to GPU if available, otherwise use CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define paths for saving output and model
model_path = './checkpoint.pth'

# Load the pretrained model
model = MultimodalNetwork(tabular_data_size=65, n_classes=3).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()  # Set the model to evaluation mode

# Function to pad sequences to the same size for batch processing
def pad_sequence(sequences):
    max_size = max([s.size() for s in sequences])
    padded_sequences = [
        F.pad(s, (0, max_size[2] - s.size(2), 0, max_size[1] - s.size(1), 0, max_size[0] - s.size(0)))
        for s in sequences
    ]
    return torch.stack(padded_sequences)

# Custom collate function to handle different data types and padding
def my_collate_fn(batch):
    batch = {k: [d[k] for d in batch] for k in batch[0]}
    for k in batch:
        if k == 'image_data':
            batch[k] = pad_sequence(batch[k])
        else:
            batch[k] = default_collate(batch[k])
    return batch

# Class to handle GradCAM (Gradient-weighted Class Activation Mapping) for visual explanations
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        # Register hooks to capture gradients and activations
        target_layer.register_forward_hook(self.save_activations)
        target_layer.register_backward_hook(self.save_gradients)

    # Hook to save activations
    def save_activations(self, module, input, output):
        self.activations = output

    # Hook to save gradients
    def save_gradients(self, module, input, output):
        self.gradients = output[0]

    # Generate CAM (Class Activation Map) for visualizing model's focus on input
    def generate(self, image_data, tabular_data, genetic_data, labels):
        self.model.eval()
        # Forward pass through the model
        output = self.model(tabular_data=tabular_data, genetic_data=genetic_data, image_data=image_data, labels=labels)[-1]
        output = F.softmax(output, dim=1)  # Apply softmax to get probabilities
        output_np = np.round(output.cpu().detach().numpy(), 4)  # Convert to numpy array and round values
        
        softmax = nn.Softmax(dim=1)
        output_softmax = softmax(output)
        pred = output_softmax.argmax(dim=1)  # Get the predicted class index
        
        # Backward pass to get gradients for the predicted class
        output[:, pred].backward()
        gradients = self.gradients[0]
        activations = self.activations[0]
        weights = torch.mean(gradients, dim=(1, 2, 3), keepdim=True)
        cam = torch.sum(weights * activations, dim=1)
        cam = F.relu(cam)  # Apply ReLU to remove negative values
        cam -= cam.min()
        cam /= cam.max()  # Normalize to range [0, 1]
        
        return cam, pred, output_np

# Configuration for the language model
max_seq_length = 4000
dtype = torch.float16
load_in_4bit = True

# Prompt template for the language model
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

# Load the language model and tokenizer
model_llm, tokenizer = FastLanguageModel.from_pretrained(
    model_name="./lora_model",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit
)

EOS_TOKEN = tokenizer.eos_token

# Function to format prompts for the language model
def formatting_prompts_func(examples):
    texts = [alpaca_prompt.format(inst, inp, out) + EOS_TOKEN for inst, inp, out in zip(examples["Instruction"], examples["Input"], examples["Output"])]
    return {"text": texts}

# Function to extract the response from the model's output
def extract_response(text):
    response_part = text.split("###")[-1]
    response = response_part[12:-17]
    return response

# Function to trim the response to the last sentence
def trim_to_last_sentence(text):
    end_punctuation = ['.', '!', '?']
    text = text.rstrip()
    last_positions = [text.rfind(punct) for punct in end_punctuation]
    last_sentence_end = max(last_positions)
    return text[:last_sentence_end + 1] if last_sentence_end != -1 else text

# Function to parse probabilities from a string
def parse_probabilities(probability_string):
    prob_dict = {}
    lines = probability_string.split('\n')
    for line in lines:
        class_name, prob = line.split(': ')
        prob_dict[class_name] = float(prob.strip('%'))
    return prob_dict

# Function to get the class with the highest probability
def get_best_class(prob_dict):
    best_class = max(prob_dict, key=prob_dict.get)
    best_prob = prob_dict[best_class]
    return best_class, best_prob

# Function to find top matches based on the best class and probability
def find_top_matches(csv_inputs, csv_outputs, best_class, best_prob, tolerance=0.5, top_k=10):
    class_mapping = {
        'Dementia': "Alzheimer's Disease",
        'CN': "Cognitive Normal",
        'MCI': "Mild Cognitive Impairment"
    }
    best_dict_class = class_mapping.get(best_class, best_class)
    matches = [(index, prob_dict, csv_outputs[index])
               for index, prob_dict in enumerate(csv_inputs)
               if abs(prob_dict.get(best_dict_class, 0) - best_prob) <= tolerance]
    matches_sorted = sorted(matches, key=lambda x: abs(x[1].get(best_dict_class, 0) - best_prob))
    return matches_sorted[:top_k]

# Define dataset and dataloader
csv_file_path = 'ehr.csv'
img_folder_path = 'images_folder'
genetic_folder_path = 'snp_folder'

dataset = MultimodalDataset(
    csv_file=csv_file_path,
    img_folder=img_folder_path,
    genetic_folder_path=genetic_folder_path,
    transform=PreprocessTransform((64, 128, 128)),
    imageuids=[73903]  # Example UID for demonstration
)

test_loader = DataLoader(
    dataset, batch_size=1, shuffle=False, collate_fn=my_collate_fn, pin_memory=True, num_workers=16
)

# Load CSV data for comparison
csv_file_path = './ADMR_test.csv'
csv_data = pd.read_csv(csv_file_path)
csv_inputs = csv_data['Input'].apply(parse_probabilities)  # Parse probabilities from CSV
csv_outputs = csv_data['Output']

# Initialize lists to store evaluation metrics
best_perplexities, best_bleu_scores, best_bertscore_p, best_bertscore_r, best_bertscore_f1 = [], [], [], [], []
best_distinct1_scores, best_distinct2_scores, best_moverscore_scores, best_repetition_rates, best_length_ratios = [], [], [], [], []

# Process each batch of data
for batch in test_loader:
    # Extract data from the batch
    tabular_data, image_data, genetic_data, labels, img_path, genetic_data_path, tabular_row = (
        batch['tabular_data'][0].unsqueeze(0),
        batch['image_data'][0].unsqueeze(0),
        batch['genetic_data'][0].unsqueeze(0),
        batch['label'][0].unsqueeze(0),
        batch['img_path'][0],
        batch['genetic_data_path'][0],
        batch['tabular_row'][0]
    )

    # Use the model to make predictions
    with torch.no_grad():
        output = model(tabular_data=tabular_data, genetic_data=genetic_data, image_data=image_data, labels=labels)[-1]
    
    # Process the output for GradCAM and SHAP
    grad_cam = GradCAM(model, model.conv2)
    cam, pred, output_np = grad_cam.generate(image_data, tabular_data, genetic_data, labels)
    
    # Format the prompt for the language model
    examples = {
        "Instruction": [f"Explain the prediction for class {i}." for i in range(output_np.shape[1])],
        "Input": [str(output_np[0])]*output_np.shape[1],
        "Output": [f"Class {i} explanation." for i in range(output_np.shape[1])]
    }
    prompts = formatting_prompts_func(examples)
    
    # Generate responses using the language model
    inputs = tokenizer(prompts["text"], return_tensors="pt", padding=True, truncation=True)
    outputs = model_llm.generate(**inputs)
    responses = [extract_response(tokenizer.decode(output, skip_special_tokens=True)) for output in outputs]
    
    # Trim the responses to the last sentence
    responses_trimmed = [trim_to_last_sentence(response) for response in responses]
    
    # Find the best class and probability
    prob_dict = parse_probabilities(responses_trimmed[0])
    best_class, best_prob = get_best_class(prob_dict)
    
    # Find top matches based on best class and probability
    top_matches = find_top_matches(csv_inputs, csv_outputs, best_class, best_prob)
    
    # Save the generated report
    output_file_path = f"generated_report_{tabular_row}.txt"
    with open(output_file_path, 'w') as f:
        f.write(f"Generated Report:\n{responses_trimmed[0]}\n\nTop Matches:\n")
        for index, prob_dict, output in top_matches:
            f.write(f"Match {index} - {prob_dict} - {output}\n")
    
    print(f"Generated report saved to {output_file_path}")

print("Processing complete.")
