import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from model import LLMActionRerankerGenerative
from config import TrainingConfig

class EarlyStopping:
    """Early stopping to prevent overfitting"""
    
    def __init__(self, patience: int = 3, min_delta: float = 0.001, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'min':
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


def compute_validation_loss(
    model,
    val_loader: DataLoader,
    device: torch.device
) -> float:
    """
    Compute validation loss WITHOUT generation (fast, parallel).
    This is used for checkpointing during training.
    """
    was_training = model.training
    model.eval()
    total_loss = 0
    num_batches = 0
    
    pbar = tqdm(val_loader, desc="Validating", leave=False)
    with torch.inference_mode():
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            
            # Count non-masked tokens for proper averaging
            total_loss += outputs.loss.item()
            num_batches += 1
            
            avg_loss = total_loss / num_batches
            pbar.set_postfix(loss=f"{outputs.loss.item():.4f}", avg=f"{avg_loss:.4f}")
            # if num_batches == 10:
            #     break # Remove this line to train on full epoch
    if was_training:
        model.train()
    return total_loss / num_batches if num_batches > 0 else float('inf')


def train_with_regularization(
    model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainingConfig,
    formatter,
    output_dir: str = "./checkpoints"
):
    """
    Improved training loop with:
    1. Validation LOSS for checkpointing (fast)
    2. Label smoothing
    3. Gradient accumulation
    4. Early stopping
    5. Generation eval only at the end
    """
    
    os.makedirs(output_dir, exist_ok=True)
    device = next(model.parameters()).device
    
    # Setup optimizer with weight decay
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Learning rate scheduler with warmup
    num_training_steps = len(train_loader) * config.num_epochs // config.gradient_accumulation_steps
    num_warmup_steps = int(num_training_steps * config.warmup_ratio)
    
    def lr_lambda(step):
        if step < num_warmup_steps:
            return step / max(1, num_warmup_steps)
        return max(0.1, 1 - (step - num_warmup_steps) / (num_training_steps - num_warmup_steps))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=config.patience,
        min_delta=config.min_delta,
        mode='min'  # We want to minimize loss
    )
    
    # Training tracking
    best_val_loss = float('inf')
    training_history = {
        'train_loss': [],
        'val_loss': [],
        'learning_rate': []
    }
    
    print("\n" + "="*70)
    print("TRAINING WITH REGULARIZATION")
    print("="*70)
    print(f"Weight decay: {config.weight_decay}")
    print(f"Label smoothing: {config.label_smoothing}")
    print(f"LoRA dropout: {config.lora_dropout}")
    print(f"Gradient accumulation steps: {config.gradient_accumulation_steps}")
    print(f"Early stopping patience: {config.patience}")
    print(f"Evaluation method: Validation LOSS (fast)")
    print("="*70 + "\n")
    
    global_step = 0
    
    for epoch in range(config.num_epochs):
        model.to(device)
        model.train()
        total_loss = 0
        num_batches = 0
        optimizer.zero_grad()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs}")
        
        for batch_idx, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            
            # Apply label smoothing manually if needed
            # (HuggingFace doesn't support it directly for LM)
            loss = outputs.loss # / config.gradient_accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Gradient accumulation
            # if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                # Clip gradients
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 
                config.max_grad_norm
            )
            
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1
                
            total_loss += outputs.loss.item()
            num_batches += 1
            
            pbar.set_postfix({
                "loss": f"{outputs.loss.item():.4f}",
                "avg": f"{total_loss/num_batches:.4f}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}"
            })
            # if num_batches == 10:
            #     break # Remove this line to train on full epoch
        # Compute validation LOSS (fast, parallel)
        val_loss = compute_validation_loss(model, val_loader, device)
        train_loss = total_loss / num_batches
        
        # Log
        training_history['train_loss'].append(train_loss)
        training_history['val_loss'].append(val_loss)
        training_history['learning_rate'].append(scheduler.get_last_lr()[0])
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  Gap:        {val_loss - train_loss:+.4f} (positive = potential overfitting)")        
        # Save best model based on validation LOSS
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'train_loss': train_loss,
                'config': config
            }, f"{output_dir}/best_model.pth")
            print(f"  ✓ New best model saved! (val_loss: {val_loss:.4f})")
        
        # Early stopping check
        if early_stopping(val_loss):
            print(f"\n⚠️ Early stopping triggered at epoch {epoch+1}")
            break
        
        print()
    
    # Save training history
    with open(f"{output_dir}/training_history.json", 'w') as f:
        json.dump(training_history, f, indent=2)
    
    # Final generation-based evaluation (only once at the end)
    if config.final_generation_eval:
        print("\n" + "="*70)
        print("FINAL GENERATION-BASED EVALUATION")
        print("="*70)
        
        # Load best model
        checkpoint = torch.load(f"{output_dir}/best_model.pth")
        model.load_state_dict(checkpoint['model_state_dict'])
        
        final_acc = evaluate_with_generation(model, val_loader, formatter, device)
        print(f"\n🎯 Final Validation Accuracy: {final_acc*100:.2f}%")
        
        # Save final results
        with open(f"{output_dir}/final_results.json", 'w') as f:
            json.dump({
                'final_accuracy': final_acc,
                'best_val_loss': best_val_loss,
                'epochs_trained': epoch + 1
            }, f, indent=2)
        
        return final_acc
    
    return best_val_loss


@torch.no_grad()
def evaluate_with_generation(
    model,
    dataloader: DataLoader,
    formatter,
    device: torch.device,
    max_samples: int = None
):
    """Generation-based evaluation (slow but accurate)"""
    model.eval()
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc="Generation Eval")
    
    for batch in pbar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        true_labels = batch["true_label"]
        prompt_lengths = batch["prompt_length"]
        
        for i in range(input_ids.shape[0]):
            prompt_len = prompt_lengths[i].item()
            prompt_ids = input_ids[i, :prompt_len].unsqueeze(0)
            prompt_mask = attention_mask[i, :prompt_len].unsqueeze(0)
            
            response = model.generate_with_reasoning(
                prompt_ids, prompt_mask, max_new_tokens=50  # Reduced for speed
            )
            
            pred_idx, _ = model.extract_action_from_response(
                response, formatter.action_names
            )
            
            correct += int(pred_idx == true_labels[i].item())
            total += 1
            
            pbar.set_postfix({"acc": f"{correct/total*100:.1f}%"})
            
            if max_samples and total >= max_samples:
                break
        
        if max_samples and total >= max_samples:
            break
    
    return correct / total
