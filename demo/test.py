import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Function
from model_v2 import MultimodalNetwork
from dataset import MultimodalDataset, PreprocessTransform
from torch.utils.data import DataLoader, random_split, TensorDataset
from torch.utils.data.dataloader import default_collate
import os
import pandas as pd
import torch.nn.functional as F
import csv
import torch.nn as nn
import re
import warnings
warnings.simplefilter('ignore')
import sys
import time
import contextlib

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

from unsloth import FastLanguageModel
# Suppress output using contextlib

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

# Grad-CAM implementation
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
        # print("=====><=====")
        # print("Probabilities:", output)
        # print("labels:",labels)
        # print("=====><=====")
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


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
max_seq_length = 4000 # Choose any! We auto support RoPE Scaling internally!
dtype = torch.float16 # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

# model_llm, tokenizer = FastLanguageModel.from_pretrained(
#     model_name = "/DATA1/arkaprabha/LLM/mistral-7b_unsloth",
#     max_seq_length = max_seq_length,
#     dtype = dtype,
#     load_in_4bit = load_in_4bit,
# )
'''
model_llm = FastLanguageModel.get_peft_model(
    model_llm,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)
'''

# print("coming?")

# Suppress output using contextlib
with open(os.devnull, 'w') as fnull:
    with contextlib.redirect_stdout(fnull), contextlib.redirect_stderr(fnull):
        model_llm, tokenizer = FastLanguageModel.from_pretrained(
            model_name = "lora_model_llama3",
            max_seq_length = max_seq_length,
            dtype = dtype,
            load_in_4bit = load_in_4bit,
        )

# print("line 161")
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

FastLanguageModel.for_inference(model_llm)

# Define the dataset and dataloader
csv_file_path = 'ehr.csv'
img_folder_path = 'images_folder'
genetic_folder_path = 'snp_folder'

dataset = MultimodalDataset(csv_file=csv_file_path,
                            img_folder=img_folder_path,
                            genetic_folder_path=genetic_folder_path,
                            transform=PreprocessTransform((64, 128, 128)),
                            imageuids=[142353])


#73903, 108440, 59739, 167021, 139021, 766218, 73037, 45943, 64323, 67128, 86261, 398183, 432875, 69219, 204345, 62636, 123708, 416028, 64663, 65134, 291872, 134505, 223365, 112391, 67668, 62198, 72838, 200229, 65498, 65013, 106612, 130245, 67522, 139235, 377081, 349639, 74321, 223517, 222569, 379930, 384092, 119520, 119510, 351355, 222797, 250641, 171854, 91985, 67743, 80894, 71669, 47935, 137280, 391070, 47703, 124467, 254815, 118777, 204403, 130179, 159987, 203683, 94946, 149510, 474208, 118996, 118713, 69501, 557537, 134207, 118990, 374493, 119182, 160356, 108237, 73145, 79796, 118979, 108098, 149553, 75073, 349864, 74694, 223257, 119190, 75128, 439316, 119202, 35629, 374561, 122981, 80953, 147406, 34159, 129842, 165298, 362989, 416067, 118903
                            
test_loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=my_collate_fn, pin_memory=True, num_workers=16)

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

    
    # print("preds:", formatted_output)

    csv_file_path = 'sample.csv'

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
    ], return_tensors="pt").to("cuda:0")

    outputs_llm = model_llm.generate(**inputs, max_new_tokens=4000, use_cache=True)

    response = extract_response(str(tokenizer.batch_decode(outputs_llm)))

    response = trim_to_last_sentence(response)

    # Assume 'formatted_output' and 'response' are already generated in your code
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

    demantia = torch.tensor([[0., 0., 1.]], device='cuda:0')
    mci = torch.tensor([[0., 1., 0.]], device='cuda:0')
    cn = torch.tensor([[1., 0., 0.]], device='cuda:0')

    if torch.equal(labels, demantia):
        label = 'Demantia'
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
        "Mild Cognitive Impairment":outputs[0][1].item(),
        "Dementia": outputs[0][2].item(),
        "LLM Response": response
    }

    file_exists = os.path.isfile(csv_file_path)

    with open(csv_file_path, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=row_data.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_data)

