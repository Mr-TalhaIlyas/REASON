"""
Proper Save/Load for QLoRA Models (4-bit + LoRA)
=================================================

1. SAVE: Only the LoRA adapter weights (not the quantized base model)
2. LOAD: Reload base model fresh, then load LoRA adapter on top

This module provides correct save/load utilities.
"""

import os
import json
import torch
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, asdict

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, LoraConfig, get_peft_model, prepare_model_for_kbit_training


# =============================================================================
#                              SAVING
# =============================================================================

def save_lora_checkpoint(
    model,  # PeftModel (your trained model)
    tokenizer,
    save_dir: str,
    config: Any = None,
    training_state: Dict = None,
    save_full_model: bool = False
):
    """
    Save a QLoRA model checkpoint properly.
    
    This saves:
    1. LoRA adapter weights (small, ~50-100MB)
    2. Tokenizer (for convenience)
    3. Training config and state (optional)
    
    Args:
        model: The PeftModel (LoRA-wrapped model)
        tokenizer: The tokenizer
        save_dir: Directory to save checkpoint
        config: Your training config (will be saved as JSON)
        training_state: Dict with epoch, optimizer state, etc.
        save_full_model: If True, also merge and save full model (large!)
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # 1. Save LoRA adapter weights using PEFT's method
    # This automatically saves:
    #   - adapter_model.safetensors (or .bin)
    #   - adapter_config.json
    model.save_pretrained(save_path)
    print(f"✓ Saved LoRA adapter to {save_path}")
    
    # 2. Save tokenizer
    tokenizer.save_pretrained(save_path)
    print(f"✓ Saved tokenizer to {save_path}")
    
    # 3. Save training config
    if config is not None:
        config_dict = asdict(config) if hasattr(config, '__dataclass_fields__') else config
        with open(save_path / "training_config.json", 'w') as f:
            json.dump(config_dict, f, indent=2, default=str)
        print(f"✓ Saved training config")
    
    # 4. Save training state (epoch, loss, etc.) - but NOT optimizer state for QLoRA
    if training_state is not None:
        # Don't save optimizer state - it's tied to the specific model instance
        state_to_save = {k: v for k, v in training_state.items() 
                        if k != 'optimizer_state_dict'}
        torch.save(state_to_save, save_path / "training_state.pt")
        print(f"✓ Saved training state")
    
    # 5. Optionally save merged full model (for easier inference, but large)
    if save_full_model:
        merged_path = save_path / "merged_model"
        print(f"Merging LoRA weights into base model...")
        merged_model = model.merge_and_unload()
        merged_model.save_pretrained(merged_path)
        tokenizer.save_pretrained(merged_path)
        print(f"✓ Saved merged model to {merged_path}")
    
    print(f"\n📁 Checkpoint saved to: {save_path}")
    return save_path


def save_training_checkpoint(
    model,
    tokenizer,
    optimizer,
    scheduler,
    epoch: int,
    global_step: int,
    train_loss: float,
    val_loss: float,
    config,
    save_dir: str,
    is_best: bool = False
):
    """
    Complete training checkpoint save (for resuming training).
    
    Note: Optimizer state is saved separately and may not work perfectly
    when resuming with QLoRA due to quantization. Best practice is to
    restart optimizer from scratch if resuming.
    """
    save_path = Path(save_dir)
    
    # Save LoRA weights
    save_lora_checkpoint(
        model=model,
        tokenizer=tokenizer,
        save_dir=save_dir,
        config=config,
        training_state={
            'epoch': epoch,
            'global_step': global_step,
            'train_loss': train_loss,
            'val_loss': val_loss,
        }
    )
    
    # Save optimizer and scheduler state separately (may not be portable)
    optim_path = save_path / "optimizer_scheduler.pt"
    torch.save({
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
    }, optim_path)
    print(f"✓ Saved optimizer/scheduler state (note: may need fresh optimizer on resume)")
    
    # Mark as best if needed
    if is_best:
        with open(save_path / "BEST_MODEL", 'w') as f:
            f.write(f"epoch={epoch}, val_loss={val_loss:.6f}")
    
    return save_path


# =============================================================================
#                              LOADING
# =============================================================================

def load_lora_model(
    checkpoint_dir: str,
    base_model_name: Optional[str] = None,
    device_map: str = "auto",
    use_4bit: bool = True,
    torch_dtype=torch.bfloat16
):
    """
    Load a QLoRA model from checkpoint.
    
    This:
    1. Loads the base model fresh (with quantization)
    2. Loads the LoRA adapter weights on top
    
    Args:
        checkpoint_dir: Directory containing adapter_model.safetensors and adapter_config.json
        base_model_name: Base model name (if None, reads from adapter_config.json)
        device_map: Device mapping strategy
        use_4bit: Whether to use 4-bit quantization
        torch_dtype: Data type for computations
    
    Returns:
        model: Loaded PeftModel
        tokenizer: Loaded tokenizer
        config: Training config (if saved)
    """
    checkpoint_path = Path(checkpoint_dir)
    
    # 1. Read adapter config to get base model name
    adapter_config_path = checkpoint_path / "adapter_config.json"
    if adapter_config_path.exists():
        with open(adapter_config_path, 'r') as f:
            adapter_config = json.load(f)
        if base_model_name is None:
            base_model_name = adapter_config.get('base_model_name_or_path')
    
    if base_model_name is None:
        raise ValueError("base_model_name must be provided or present in adapter_config.json")
    
    print(f"Loading base model: {base_model_name}")
    
    # 2. Setup quantization config
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=True
        )
    else:
        bnb_config = None
    
    # 3. Load base model fresh
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map=device_map,
        trust_remote_code=True,
        torch_dtype=torch_dtype
    )
    print(f"✓ Loaded base model")
    
    # 4. Load LoRA adapter on top
    model = PeftModel.from_pretrained(
        base_model,
        checkpoint_path,
        is_trainable=False  # Set True if you want to continue training
    )
    print(f"✓ Loaded LoRA adapter from {checkpoint_path}")
    
    # 5. Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"✓ Loaded tokenizer")
    
    # 6. Load training config if exists
    config = None
    config_path = checkpoint_path / "training_config.json"
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"✓ Loaded training config")
    
    return model, tokenizer, config


def load_for_inference(
    checkpoint_dir: str,
    base_model_name: Optional[str] = None,
    merge_weights: bool = True,
    device_map: str = "auto"
):
    """
    Load model optimized for inference.
    
    Args:
        checkpoint_dir: Path to checkpoint
        base_model_name: Base model name
        merge_weights: If True, merge LoRA into base (faster inference, more memory)
        device_map: Device mapping
    
    Returns:
        model: Ready for inference
        tokenizer: Tokenizer
    """
    model, tokenizer, config = load_lora_model(
        checkpoint_dir=checkpoint_dir,
        base_model_name=base_model_name,
        device_map=device_map
    )
    
    if merge_weights:
        print("Merging LoRA weights into base model...")
        model = model.merge_and_unload()
        print("✓ Weights merged (faster inference)")
    
    model.eval()
    return model, tokenizer


def load_for_continued_training(
    checkpoint_dir: str,
    base_model_name: Optional[str] = None,
    device_map: str = "auto",
    use_4bit: bool = True
):
    """
    Load model for continued training.
    
    Note: Optimizer state may not be fully compatible. Consider starting
    with a fresh optimizer for best results.
    """
    checkpoint_path = Path(checkpoint_dir)
    
    # Load model with LoRA trainable
    model, tokenizer, config = load_lora_model(
        checkpoint_dir=checkpoint_dir,
        base_model_name=base_model_name,
        device_map=device_map,
        use_4bit=use_4bit
    )
    
    # Make LoRA layers trainable again
    for name, param in model.named_parameters():
        if 'lora' in name.lower():
            param.requires_grad = True
    
    # Load training state
    training_state = None
    state_path = checkpoint_path / "training_state.pt"
    if state_path.exists():
        training_state = torch.load(state_path)
        print(f"✓ Loaded training state: epoch={training_state.get('epoch')}")
    
    return model, tokenizer, config, training_state

