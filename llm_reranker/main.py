#%%
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from torch.utils.data import DataLoader

from data import (ConceptFormatter, RegularizedGenerativeDataset,
                    analyze_distribution_shift, create_mixed_dataset)
from config import TrainingConfig
from model import LLMActionRerankerGenerative
from tools import train_with_regularization

config = TrainingConfig()
# Initialize formatter with your concept CSV
formatter = ConceptFormatter("/data_hdd/talha/nsai/scal/llm/data/ntu_rgbd_spatial_temporal.csv")

# Assume you've extracted predictions from your skeleton model
# concept_preds: [N, 80], action_preds: [N, 120], labels: [N]
print("Loading extracted predictions...")

# Example: Load from saved numpy files
test_concept_preds = np.load("/data_hdd/talha/nsai/scal/llm/data/test_concept_probs.npy")
test_action_preds = np.load("/data_hdd/talha/nsai/scal/llm/data/test_action_probs.npy")
test_labels = np.load("/data_hdd/talha/nsai/scal/llm/data/test_action_labels.npy")

train_concept_preds = np.load("/data_hdd/talha/nsai/scal/llm/data/train_concept_probs.npy")
train_action_preds = np.load("/data_hdd/talha/nsai/scal/llm/data/train_action_probs.npy")
train_labels = np.load("/data_hdd/talha/nsai/scal/llm/data/train_action_labels.npy")

N = train_concept_preds.shape[0]


# _ = analyze_distribution_shift(train_concept_preds, test_concept_preds, formatter.all_concepts)
# train_concept_preds, train_action_preds, train_labels = \
#     create_mixed_dataset(train_concept_preds, train_action_preds, train_labels,
#                          test_concept_preds, test_action_preds, test_labels,
#                          config.mix_val_samples, config.mix_ratio)


# Initialize model
print("Loading LLM reranker...")
model = LLMActionRerankerGenerative(config)

# Create datasets
train_dataset = RegularizedGenerativeDataset(
    train_concept_preds,
    train_action_preds,
    train_labels,
    formatter,
    model.tokenizer,
    config,
    True
)

val_dataset = RegularizedGenerativeDataset(
    test_concept_preds,
    test_action_preds,
    test_labels,
    formatter,
    model.tokenizer,
    config,
    False
)


train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

#%%

print("Starting training...")
best_val_loss = train_with_regularization(
    model, 
    train_loader, 
    val_loader, 
    config, 
    formatter,
    output_dir="./checkpoints",
    # Wandb settings
    use_wandb=True,
    wandb_project="llm-action-reranker",
    wandb_run_name=f"lora-r{config.lora_r}-lr{config.learning_rate}",
    wandb_tags=["llama-3.1", "lora", "ntu-rgbd"],
    wandb_notes="LLM action reranker with concept bottleneck",
    log_every_n_steps=10
)
print(f"Training complete! Best validation loss: {best_val_loss:.4f}")
# print("Starting training...")
# best_val_loss = train_with_regularization(model, train_loader, val_loader, config, formatter)
# print(f"Training complete! Best validation loss: {best_val_loss:.4f}")



