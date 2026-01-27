import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠️ wandb not installed. Run: pip install wandb")

from model import LLMActionRerankerGenerative
from config import TrainingConfig
from chkpt_utils import save_lora_checkpoint

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


def init_wandb(
    config: TrainingConfig,
    project_name: str = "llm-action-reranker",
    run_name: str = None,
    tags: List[str] = None,
    notes: str = None
):
    """Initialize wandb logging"""
    if not WANDB_AVAILABLE:
        print("⚠️ wandb not available, skipping initialization")
        return None
    
    # Convert config to dict
    config_dict = asdict(config) if hasattr(config, '__dataclass_fields__') else vars(config)
    
    run = wandb.init(
        project=project_name,
        name=run_name,
        config=config_dict,
        tags=tags or ["lora", "llama", "action-recognition"],
        notes=notes,
        reinit=True
    )
    
    print(f"📊 Wandb initialized: {run.url}")
    return run


def compute_validation_loss(
    model,
    val_loader: DataLoader,
    device: torch.device,
    log_to_wandb: bool = False,
    global_step: int = None
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
            
            total_loss += outputs.loss.item()
            num_batches += 1
            
            avg_loss = total_loss / num_batches
            pbar.set_postfix(loss=f"{outputs.loss.item():.4f}", avg=f"{avg_loss:.4f}")
            # if num_batches == 3:
            #     break # for debugging purposes only
    if was_training:
        model.train()
    
    final_loss = total_loss / num_batches if num_batches > 0 else float('inf')
    
    return final_loss


def train_with_regularization(
    model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainingConfig,
    formatter,
    output_dir: str = "./checkpoints",
    # Wandb settings
    use_wandb: bool = True,
    wandb_project: str = "llm-action-reranker",
    wandb_run_name: str = None,
    wandb_tags: List[str] = None,
    wandb_notes: str = None,
    log_every_n_steps: int = 10
):
    """
    Improved training loop with:
    1. Validation LOSS for checkpointing (fast)
    2. Label smoothing
    3. Gradient accumulation
    4. Early stopping
    5. Generation eval only at the end
    6. Wandb logging
    """
    
    os.makedirs(output_dir, exist_ok=True)
    device = next(model.parameters()).device
    
    # Initialize wandb
    wandb_run = None
    if use_wandb and WANDB_AVAILABLE:
        wandb_run = init_wandb(
            config=config,
            project_name=wandb_project,
            run_name=wandb_run_name,
            tags=wandb_tags,
            notes=wandb_notes
        )
        
        # Log model architecture info
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        wandb.log({
            "model/trainable_params": trainable_params,
            "model/total_params": total_params,
            "model/trainable_ratio": trainable_params / total_params,
            "data/train_samples": len(train_loader.dataset),
            "data/val_samples": len(val_loader.dataset),
            "data/train_batches": len(train_loader),
            "data/val_batches": len(val_loader),
        })
    
    # Setup optimizer with weight decay
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Learning rate scheduler with warmup
    num_training_steps = len(train_loader) * config.num_epochs
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
        mode='min'
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
    print(f"Wandb logging: {'Enabled' if wandb_run else 'Disabled'}")
    print("="*70 + "\n")
    
    global_step = 0
    
    for epoch in range(config.num_epochs):
        model.to(device)
        model.train()
        total_loss = 0
        num_batches = 0
        epoch_losses = []  # Track individual batch losses for statistics
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
            
            loss = outputs.loss
            
            # Backward pass
            loss.backward()
            
            # Clip gradients
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), 
                config.max_grad_norm
            )
            
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1
            
            batch_loss = outputs.loss.item()
            total_loss += batch_loss
            num_batches += 1
            epoch_losses.append(batch_loss)
            
            # Log to wandb every N steps
            if wandb_run and global_step % log_every_n_steps == 0:
                wandb.log({
                    "train/batch_loss": batch_loss,
                    "train/running_avg_loss": total_loss / num_batches,
                    "train/learning_rate": scheduler.get_last_lr()[0],
                    "train/grad_norm": grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm,
                    "train/epoch": epoch + 1,
                    "train/global_step": global_step,
                }, step=global_step)
            
            pbar.set_postfix({
                "loss": f"{batch_loss:.4f}",
                "avg": f"{total_loss/num_batches:.4f}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}"
            })
            # if num_batches == 3:
            #     break # for debugging purposes only
        # Compute epoch statistics
        train_loss = total_loss / num_batches
        train_loss_std = np.std(epoch_losses)
        train_loss_min = min(epoch_losses)
        train_loss_max = max(epoch_losses)
        
        # Compute validation loss
        val_loss = compute_validation_loss(model, val_loader, device)
        
        # Calculate gap
        loss_gap = val_loss - train_loss
        
        # Log epoch metrics
        training_history['train_loss'].append(train_loss)
        training_history['val_loss'].append(val_loss)
        training_history['learning_rate'].append(scheduler.get_last_lr()[0])
        
        # Log to wandb
        if wandb_run:
            wandb.log({
                # Epoch-level training metrics
                "epoch/train_loss": train_loss,
                "epoch/train_loss_std": train_loss_std,
                "epoch/train_loss_min": train_loss_min,
                "epoch/train_loss_max": train_loss_max,
                
                # Validation metrics
                "epoch/val_loss": val_loss,
                
                # Gap and overfitting indicators
                "epoch/loss_gap": loss_gap,
                "epoch/overfitting_ratio": val_loss / train_loss if train_loss > 0 else 0,
                
                # Best metrics so far
                "epoch/best_val_loss": min(best_val_loss, val_loss),
                
                # Training progress
                "epoch/epoch": epoch + 1,
                "epoch/learning_rate": scheduler.get_last_lr()[0],
                "epoch/early_stopping_counter": early_stopping.counter,
            }, step=global_step)
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_loss:.4f} (±{train_loss_std:.4f})")
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  Gap:        {loss_gap:+.4f} (positive = potential overfitting)")
        
        # Save best model based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            
            save_lora_checkpoint(
                model=model.model,
                tokenizer=model.tokenizer,
                save_dir=str( f"{output_dir}/best_model"),
                config=config,
                training_state={
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }
            )
            print(f"✓ New best model saved!")
            print(f"  ✓ New best model saved! (val_loss: {val_loss:.4f})")
            
            # Log model artifact to wandb
            if wandb_run:
                wandb.run.summary["best_val_loss"] = val_loss
                wandb.run.summary["best_epoch"] = epoch + 1
        
        save_lora_checkpoint(
            model=model.model,
            tokenizer=model.tokenizer,
            save_dir=str(f"{output_dir}/latest"),
            config=config,
            training_state={
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
            }
        )
        # Early stopping check
        if early_stopping(val_loss):
            print(f"\n⚠️ Early stopping triggered at epoch {epoch+1}")
            if wandb_run:
                wandb.log({"training/early_stopped": True, "training/stopped_epoch": epoch + 1})
            break
        
        print()
    
    # Save training history
    with open(f"{output_dir}/training_history.json", 'w') as f:
        json.dump(training_history, f, indent=2)
    
    # Log final summary to wandb
    if wandb_run:
        wandb.run.summary["final_train_loss"] = train_loss
        wandb.run.summary["final_val_loss"] = val_loss
        wandb.run.summary["total_epochs"] = epoch + 1
        wandb.run.summary["total_steps"] = global_step
        
        # Save training history as artifact
        artifact = wandb.Artifact("training_history", type="results")
        artifact.add_file(f"{output_dir}/training_history.json")
        wandb.log_artifact(artifact)
    
    # Final generation-based evaluation (only once at the end)
    if config.final_generation_eval:
        print("\n" + "="*70)
        print("FINAL GENERATION-BASED EVALUATION")
        print("="*70)
        
        # Load best model
        checkpoint = torch.load(f"{output_dir}/best_model.pth")
        model.load_state_dict(checkpoint['model_state_dict'])
        
        final_acc = evaluate_with_generation(
            model, val_loader, formatter, device,
            log_to_wandb=wandb_run is not None
        )
        print(f"\n🎯 Final Validation Accuracy: {final_acc*100:.2f}%")
        
        # Log final accuracy
        if wandb_run:
            wandb.run.summary["final_accuracy"] = final_acc
            wandb.log({"eval/final_accuracy": final_acc})
        
        # Save final results
        with open(f"{output_dir}/final_results.json", 'w') as f:
            json.dump({
                'final_accuracy': final_acc,
                'best_val_loss': best_val_loss,
                'epochs_trained': epoch + 1
            }, f, indent=2)
        
        # Finish wandb run
        if wandb_run:
            wandb.finish()
        
        return final_acc
    
    # Finish wandb run
    if wandb_run:
        wandb.finish()
    
    return best_val_loss


@torch.no_grad()
def evaluate_with_generation(
    model,
    dataloader: DataLoader,
    formatter,
    device: torch.device,
    max_samples: int = None,
    log_to_wandb: bool = False
):
    """Generation-based evaluation (slow but accurate)"""
    model.eval()
    correct = 0
    total = 0
    
    # Track predictions for confusion analysis
    predictions = []
    ground_truths = []
    
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
            
            with torch.inference_mode():
                response = model.generate_with_reasoning(
                    prompt_ids, prompt_mask, max_new_tokens=50
                )
            # print model response for debugging
            # print(response)
            pred_idx, _ = model.extract_action_from_response(
                response, formatter.action_names
            )
            
            true_label = true_labels[i].item()
            is_correct = int(pred_idx == true_label)
            
            correct += is_correct
            total += 1
            
            predictions.append(pred_idx)
            ground_truths.append(true_label)
            
            pbar.set_postfix({"acc": f"{correct/total*100:.1f}%"})
            
            if max_samples and total >= max_samples:
                break
        
        if max_samples and total >= max_samples:
            break
    
    accuracy = correct / total
    
    # Log to wandb
    if log_to_wandb and WANDB_AVAILABLE:
        # Create confusion matrix
        try:
            wandb.log({
                "eval/accuracy": accuracy,
                "eval/correct": correct,
                "eval/total": total,
                "eval/confusion_matrix": wandb.plot.confusion_matrix(
                    y_true=ground_truths,
                    preds=predictions,
                    class_names=formatter.action_names[:max(max(predictions)+1, max(ground_truths)+1)] if predictions else None
                )
            })
        except Exception as e:
            print(f"⚠️ Could not log confusion matrix: {e}")
            wandb.log({
                "eval/accuracy": accuracy,
                "eval/correct": correct,
                "eval/total": total,
            })
    
    return accuracy, predictions, ground_truths