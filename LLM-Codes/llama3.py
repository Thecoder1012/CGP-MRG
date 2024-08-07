import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from transformers import TrainerCallback, TrainerState, TrainerControl
import torch
import logging
import wandb

# Set up wandb
wandb.login(key="5deddb0a12b5ef755ce536d4b3e9a55380e68e73")
wandb.init(project="LLM_Alzheimer", entity="arkaprabha17")  # Replace 'your_username' with your personal username


class LoggingCallback(TrainerCallback):
    def on_log(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        log_info = {
            "Step": state.global_step,
            "Loss": state.log_history[-1].get('loss', 'N/A'),
            "Grad Norm": state.log_history[-1].get('grad_norm', 'N/A'),
            "Learning rate": state.log_history[-1].get('learning_rate', 'N/A'),
        }
        logging.info(log_info)
        wandb.log(log_info)
'''
class LoggingCallback(TrainerCallback):
    def on_log(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        log_info = {
            "Step": state.global_step,
            "Loss": state.log_history[-1].get('loss', 'N/A'),
            "Grad Norm": state.log_history[-1].get('grad_norm', 'N/A'),
            "Learning rate": state.log_history[-1].get('learning_rate', 'N/A'),
            "Validation Loss": state.log_history[-1].get('eval_loss', 'N/A'),
            "Accuracy": state.log_history[-1].get('eval_accuracy', 'N/A'),
            "Precision": state.log_history[-1].get('eval_precision', 'N/A'),
            "Recall": state.log_history[-1].get('eval_recall', 'N/A'),
            "F1 Score": state.log_history[-1].get('eval_f1', 'N/A'),
            "Train Samples Per Second": state.log_history[-1].get('train_samples_per_second', 'N/A'),
            "Train Steps Per Second": state.log_history[-1].get('train_steps_per_second', 'N/A'),
            "Total Flops": state.log_history[-1].get('total_flos', 'N/A'),
            "Total Training Time": state.log_history[-1].get('total_training_time', 'N/A')
        }
        logging.info(log_info)
        wandb.log(log_info)
'''
# Ensure CUDA is available
if not torch.cuda.is_available():
    raise SystemError("CUDA is not available. Please check your CUDA installation.")

device = torch.device("cuda:0")
# major_version, minor_version = torch.cuda.get_device_capability(device)   

# Must install separately since Colab has torch 2.2.1, which breaks packages
'''
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

if major_version >= 8:
    # Use this for new GPUs like Ampere, Hopper GPUs (RTX 30xx, RTX 40xx, A100, H100, L40)
    !pip install --no-deps packaging ninja einops flash-attn xformers trl peft accelerate bitsandbytes
else:
    # Use this for older GPUs (V100, Tesla T4, RTX 20xx)
    !pip install --no-deps xformers trl peft accelerate bitsandbytes
pass
'''

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                    handlers=[logging.FileHandler("training_stats_llama3_15_06_2024.log"),
                              logging.StreamHandler()])

from unsloth import FastLanguageModel
import torch
max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = torch.float16 # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

# 4bit pre quantized models we support for 4x faster downloading + no OOMs.
# fourbit_models = [
#     "unsloth/mistral-7b-bnb-4bit",
#     "unsloth/mistral-7b-instruct-v0.2-bnb-4bit",
#     "unsloth/llama-2-7b-bnb-4bit",
#     "unsloth/gemma-7b-bnb-4bit",
#     "unsloth/gemma-7b-it-bnb-4bit", # Instruct version of Gemma 7b
#     "unsloth/gemma-2b-bnb-4bit",
#     "unsloth/gemma-2b-it-bnb-4bit", # Instruct version of Gemma 2b
#     "unsloth/llama-3-8b-bnb-4bit", # [NEW] 15 Trillion token Llama-3
# ] # More models at https://huggingface.co/unsloth

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-3-8b-bnb-4bit",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    cache_dir = "./llama3_8b_experiment_2",
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)

model = FastLanguageModel.get_peft_model(
    model,
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

model = model.to(device)
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Output:
{}"""

EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
def formatting_prompts_func(examples):
    instructions = examples["Instruction"]
    inputs       = examples["Input"]
    outputs      = examples["Output"]
    texts = []
    for instruction, input1, output in zip(instructions, inputs, outputs):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = alpaca_prompt.format(instruction, input1, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }
pass

from datasets import load_dataset
# dataset = load_dataset("yahma/alpaca-cleaned", split = "train")
# csv_file_path = "/content/drive/MyDrive/LLM/dataset/Alzheimers_Disease_Medical_Report_Dataset_20k_Rows.csv" #for arkaprabha1012
# csv_file_path = "./dataset/Alzheimers_Disease_Medical_Report_20000_Rows.csv" #for arkaprabha17
csv_file_path = "./dataset/detailed_medical_reports_dataset15_06_2024.csv"
dataset = load_dataset('csv', data_files={'train': csv_file_path}) #for own dataset loading
# dataset = load_dataset("yahma/alpaca-cleaned")
dataset = dataset.map(formatting_prompts_func, batched = True,)

#for own dataset training
from trl import SFTTrainer
from transformers import TrainingArguments

# Assuming `dataset` and `max_seq_length` are correctly set up
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset['train'],  # ensure this points to a dataset split
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=8,
    packing=False,
    args=TrainingArguments(
        output_dir="outputs",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=5,  # example parameter you might need
        warmup_steps=5,
        #max_steps=700,
        learning_rate=2e-4,
        fp16=torch.cuda.is_available(),  # simplified check for FP16 support
        logging_steps=1,
        optim="adamw_torch",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407
    ),
    callbacks=[LoggingCallback]  # Adding the custom logging callback here
)

#@title Show current memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
logging.info(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
logging.info(f"{start_gpu_memory} GB of memory reserved.")

trainer_stats = trainer.train()

logging.info("Training completed. Here are the stats:")
logging.info(trainer_stats)

# Log the training_stats.log file to wandb
wandb.save("training_stats_llama3_15_06_2024.log")

# alpaca_prompt = Copied from above
FastLanguageModel.for_inference(model) # Enable native 2x faster inference
inputs = tokenizer(
[
    alpaca_prompt.format(
        "Make a medical report with summary.", # instruction
        "Cognitively Normal: 99.43%, \n Mild Cognitive Impairment: 0.20% \n Alzheimer Disease: 0.33%.", # input
        "", # output - leave this blank for generation!
    )
], return_tensors = "pt").to("cuda:0")

outputs = model.generate(**inputs, max_new_tokens = 64, use_cache = True)
logging.info(tokenizer.batch_decode(outputs))

model.save_pretrained("lora_model_llama3_15_06_2024") # Local saving
# model.push_to_hub("your_name/lora_model", token = "...") # Online saving

