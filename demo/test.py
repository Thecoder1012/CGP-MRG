import os
import re
import csv
import time
import torch
import warnings
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, default_collate
from torch.autograd import Function
from model_v2 import MultimodalNetwork
from dataset import MultimodalDataset, PreprocessTransform
from unsloth import FastLanguageModel
import contextlib
import sys

# Suppress warnings
warnings.simplefilter('ignore')

# Utility function for live printing with delay
def live_print(text, delay=0.05):
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    sys.stdout.write('\n')

# Load the pretrained model
model_path = './checkpoint.pth'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = MultimodalNetwork(tabular_data_size=65, n_classes=3).to(device)
model.load_state_dict(torch.load(model_path))
model.eval()

# Function to pad sequences for batch processing
def pad_sequence(sequences):
    max_size = max([s.size() for s in sequences])
    padded_sequences = [F.pad(s, (0, max_size[2] - s.size(2), 0, max_size[1] - s.size(1), 0, max_size[0] - s.size(0))) for s in sequences]
    return torch.stack(padded_sequences)

# Custom collate function for DataLoader
def my_collate_fn(batch):
    batch = {k: [d[k] for d in batch] for k in batch[0]}
    for k in batch:
        if k == 'image_data':
            batch[k] = pad_sequence(batch[k])
        else:
            batch[k] = default_collate(batch[k])
    return batch

# Grad-CAM implementation for visualization
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self.save_activations)
        target_layer.register_backward_hook(self.save_gradients)

    def save_activations(self, module, input, output):
        self.activations = output

    def save_gradients(self, module, input, output):
        self.gradients = output[0]

    def generate(self, image_data, tabular_data, genetic_data, labels):
        self.model.eval()
        output = self.model(tabular_data=tabular_data, genetic_data=genetic_data, image_data=image_data, labels=labels)[-1]
        output = F.softmax(output, dim=1)
        output_np = np.round(output.cpu().detach().numpy(), 4)
        softmax = nn.Softmax(dim=1)
        output_softmax = softmax(output)
        pred = output_softmax.argmax(dim=1)
        output[:, pred].backward()
        gradients = self.gradients[0]
        activations = self.activations[0]
        weights = torch.mean(gradients, dim=(1, 2, 3), keepdim=True)
        cam = torch.sum(weights * activations, dim=1)
        cam = F.relu(cam)
        cam -= cam.min()
        cam /= cam.max()
        return cam, pred, output_np

# Set device and model parameters for FastLanguageModel
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
max_seq_length = 4000
dtype = torch.float16
load_in_4bit = True

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

# Load the FastLanguageModel with suppressed output
with open(os.devnull, 'w') as fnull:
    with contextlib.redirect_stdout(fnull), contextlib.redirect_stderr(fnull):
        model_llm, tokenizer = FastLanguageModel.from_pretrained(
            model_name="lora_model_llama3",
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
        )

EOS_TOKEN = tokenizer.eos_token

def formatting_prompts_func(examples):
    instructions = examples["Instruction"]
    inputs = examples["Input"]
    outputs = examples["Output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return {"text": texts}

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

# Prepare dataset and DataLoader
csv_file_path = 'ehr.csv'
img_folder_path = 'images_folder'
genetic_folder_path = 'snp_folder'

dataset = MultimodalDataset(csv_file=csv_file_path,
                            img_folder=img_folder_path,
                            genetic_folder_path=genetic_folder_path,
                            transform=PreprocessTransform((64, 128, 128)),
                            imageuids=[142353])

test_loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=my_collate_fn, pin_memory=True, num_workers=16)

# Iterate through the DataLoader for predictions and report generation
for batch in test_loader:
    tabular_data, image_data, genetic_data, labels, img_path, genetic_data_path, tabular_row = (
        batch['tabular_data'][0].unsqueeze(0),
        batch['image_data'][0].unsqueeze(0),
        batch['genetic_data'][0].unsqueeze(0),
        batch['label'][0].unsqueeze(0),
        batch['img_path'][0],
        batch['genetic_path'][0],
        batch['tabular_row'][0]
    )

    tabular_data, image_data, genetic_data, labels = (
        tabular_data.to(device),
        image_data.to(device),
        genetic_data.to(device),
        labels.to(device)
    )

    target_layer = model.image_branch[-3]
    grad_cam = GradCAM(model, target_layer)
    
    cam, pred, outputs = grad_cam.generate(image_data, tabular_data, genetic_data, labels)
    
    formatted_output = (
        f"Cognitively Normal: {100 * outputs[0][0]:.4f}\n"
        f"Mild Cognitive Impairment: {100 * outputs[0][1]:.4f}\n"
        f"Dementia: {100 * outputs[0][2]:.4f}"
    )

    # Prepare input for language model and generate response
    input_llm = formatted_output

    torch.cuda.empty_cache()

    inputs = tokenizer(
        [
            alpaca_prompt.format(
                "Make a medical report with summary.", 
                input_llm,
                "", 
            )
        ], return_tensors="pt").to("cuda:0")

    outputs_llm = model_llm.generate(**inputs, max_new_tokens=4000, use_cache=True)

    response = extract_response(str(tokenizer.batch_decode(outputs_llm)))
    response = trim_to_last_sentence(response)

    # Print formatted report
    live_print("Processing ImageUID: 142353")
    live_print("\n========================\n")
    live_print("Patient Report:\n")
    live_print("========================\n")
    live_print("Cognitive Status:\n")
    live_print(formatted_output)
    live_print("----------------------------")
    live_print("\nMedical Report:\n")
    live_print("----------------------------")
    live_print(response)
    live_print("\nEnd of Report.\n")

    # Define class labels
    demantia = torch.tensor([[0., 0., 1.]], device='cuda:0')
    mci = torch.tensor([[0., 1., 0.]], device='cuda:0')
    cn = torch.tensor([[1., 0., 0.]], device='cuda:0')

    if torch.equal(labels, demantia):
        label = 'Dementia'
    elif torch.equal(labels, mci):
        label = 'MCI'
    elif torch.equal(labels, cn):
        label = 'CN'

    row_data = {
        "Image Path": img_path,
        "Genetic Data Path": genetic_data_path,
        "Tabular Data": tabular_row,
        "Label": label,
        "Cognitively Normal": outputs[0][0].item(),
        "Mild Cognitive Impairment": outputs[0][1].item(),
        "Dementia": outputs[0][2].item(),
        "LLM Response": response
    }

    # Save results to CSV
    file_exists = os.path.isfile(csv_file_path)

    with open(csv_file_path, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=row_data.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_data)
