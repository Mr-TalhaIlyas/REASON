#%%
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
from torch.utils.data import DataLoader

from data import (ConceptFormatter, RegularizedGenerativeDataset,
                    analyze_distribution_shift, create_mixed_dataset)
from config import TrainingConfig
from model import InferenceModel
from tools import evaluate_with_generation
from chkpt_utils import load_lora_model

EVAL_PART_NO = 3
config = TrainingConfig(batch_size=1)
# Initialize formatter with your concept CSV
formatter = ConceptFormatter("/data_hdd/talha/nsai/scal/llm_car/data/ntu_rgbd_spatial_temporal.csv")

# Assume you've extracted predictions from your skeleton model
# concept_preds: [N, 80], action_preds: [N, 120], labels: [N]
print("Loading extracted predictions...")

# Example: Load from saved numpy files
test_concept_preds = np.load("/data_hdd/talha/nsai/scal/llm_car/data/test_concept_probs.npy")
test_action_preds = np.load("/data_hdd/talha/nsai/scal/llm_car/data/test_action_probs.npy")
test_labels = np.load("/data_hdd/talha/nsai/scal/llm_car/data/test_action_labels.npy")

N = test_concept_preds.shape[0]
print(f"Test set size: {N} samples")
split_into_parts = 8

parts_concept = np.array_split(test_concept_preds, split_into_parts)
parts_action = np.array_split(test_action_preds, split_into_parts)
parts_labels = np.array_split(test_labels, split_into_parts)
# only keep 1 part for faster evaluation
test_concept_preds = parts_concept[EVAL_PART_NO]
test_action_preds = parts_action[EVAL_PART_NO]
test_labels = parts_labels[EVAL_PART_NO]

print('Running evaluation on part:', EVAL_PART_NO)
print(f"Using {test_concept_preds.shape[0]} samples for evaluation.")
print("Concept preds shape:", test_concept_preds.shape)
print("Action preds shape:", test_action_preds.shape)
print("Labels shape:", test_labels.shape)
#%%

# Initialize model
print("Loading LLM reranker...")


peft_model, tokenizer, _ = load_lora_model(
                    checkpoint_dir="./new/best_model",
                    base_model_name=config.model_name,
                    device_map="auto",
                    use_4bit=config.use_4bit,
                    torch_dtype=torch.bfloat16
                )
model = InferenceModel(peft_model, tokenizer, config)
model.eval()

device = next(peft_model.parameters()).device

#%%
val_dataset = RegularizedGenerativeDataset(
    test_concept_preds,
    test_action_preds,
    test_labels,
    formatter,
    model.tokenizer,
    config,
    False # <- is_train
)

val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

#%%

print("Starting training...")
accuracy, predictions, ground_truths = evaluate_with_generation(
    model,
    val_loader, 
    formatter,
    device,
    max_samples = None,
    log_to_wandb = False
)
print(f"Training complete! Best accuracy: {accuracy:.4f}")
# save predictions and ground truths for further analysis
np.save("./sep_eval_predictions_part{}_acc{:.4f}.npy".format(EVAL_PART_NO, accuracy), predictions)
np.save("./sep_eval_ground_truths_part{}_acc{:.4f}.npy".format(EVAL_PART_NO, accuracy), ground_truths)
#%%