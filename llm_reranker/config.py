#%%
from dataclasses import dataclass

@dataclass
class TrainingConfig:
    """Extended config with regularization options"""
    # Model settings
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1  # Increased from 0.05
    
    top_k_actions: int = 15
    concept_threshold: float = 0.3
    
    use_4bit: bool = True
    # Training settings
    learning_rate: float = 2e-4
    batch_size: int = 1 #12
    num_epochs: int = 5
    warmup_ratio: float = 0.1
    max_seq_length: int = 1024
    max_new_tokens: int = 150  # For generating reasoning
    
    # Regularization settings (NEW)
    weight_decay: float = 0.05          # Increased from 0.01
    label_smoothing: float = 0.0        # Smooths the target distribution
    gradient_accumulation_steps: int = 2 # Effective batch size = batch_size * this
    max_grad_norm: float = 0.5          # Stricter gradient clipping
    dropout_rate: float = 0.15          # Additional dropout in classifier (if used)
    
    # Early stopping (NEW)
    patience: int = 3                   # Stop if no improvement for N epochs
    min_delta: float = 0.0001            # Minimum improvement to count
    
    # Train/Val mixing (NEW)
    mix_val_samples: bool = False       # Whether to mix val samples into train
    mix_ratio: float = 0.02              # Fraction of val to add to train
    
    # Evaluation settings
    eval_with_generation: bool = False   # Only at end, use loss during training
    final_generation_eval: bool = False   # Do generation eval at the very end
    
    # Data augmentation (NEW)
    concept_noise_std: float = 0.05      # Add Gaussian noise to concept probs
    concept_dropout: float = 0.05        # Randomly zero out some concepts