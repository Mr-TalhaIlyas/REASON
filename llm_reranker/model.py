import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from huggingface_hub import login

from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig, 
    get_peft_model, 
    prepare_model_for_kbit_training,
    TaskType
)

from config import TrainingConfig

HF_TOKEN = "...PLACE YOUR HF TOKEN HERE..." # e.g., "hf_..." 

if HF_TOKEN:
    login(token=HF_TOKEN)
    
class LLMActionRerankerGenerative(nn.Module):
    """
    Generation-based LLM re-ranker that can explain its reasoning.
    
    ARCHITECTURE:
    =============
    
    Same LLaMA backbone as classification version, but:
    - Uses the built-in language modeling head (predicts next token)
    - No additional classification head
    - Trained with causal LM loss
    
    ┌──────────────────────────────────────────────────────────────────┐
    │                    Input: Token IDs [B, L]                       │
    │   "...concepts: hand grasp... Which action? Answer:"             │
    └──────────────────────────────┬───────────────────────────────────┘
                                   │
                                   ▼
    ┌──────────────────────────────────────────────────────────────────┐
    │              LLaMA Backbone (with LoRA adapters)                 │
    │                                                                  │
    │   Embedding → Transformer Layers (×32) → LayerNorm              │
    │                                                                  │
    │   Output: hidden_states [B, L, 4096]                            │
    └──────────────────────────────┬───────────────────────────────────┘
                                   │
                                   ▼
    ┌──────────────────────────────────────────────────────────────────┐
    │                   LM Head (built-in, shared weights)             │
    │                                                                  │
    │   Linear(4096 → vocab_size)  # ~128K for LLaMA                  │
    │                                                                  │
    │   Output: logits [B, L, vocab_size]                             │
    │   (probability distribution over vocabulary for each position)   │
    └──────────────────────────────────────────────────────────────────┘
    """
    
    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.config = config
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Quantization config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True
        ) if config.use_4bit else None
        
        # Load base model
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        )
        
        if config.use_4bit:
            print('Preparing model for 4-bit training...')
            self.model = prepare_model_for_kbit_training(self.model)
        
        # Configure LoRA
        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                           "gate_proj", "up_proj", "down_proj"]
        )
        
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
    
    def forward(
        self, 
        input_ids: torch.Tensor,       # [B, L]
        attention_mask: torch.Tensor,  # [B, L]
        labels: torch.Tensor = None    # [B, L] - TOKEN-LEVEL labels, not class indices!
    ):
        """
        Forward pass for training.
        
        IMPORTANT: labels here are TOKEN-LEVEL labels for language modeling,
        NOT class indices like in ClassificationHeadReranker!
        
        labels[i, j] = expected token at position j for sample i
        labels[i, j] = -100 means ignore this position in loss (used for prompt)
        """
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
    
    @torch.no_grad()
    def generate_with_reasoning(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int = 150
    ) -> str:
        """
        Generate action prediction WITH reasoning explanation.
        
        This is where the neuro-symbolic magic happens - the LLM explains
        which concepts led to its decision.
        """
        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Greedy decoding for reproducibility
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )
        
        # Decode only the generated part (not the prompt)
        generated_tokens = outputs[0, input_ids.shape[1]:]
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        return generated_text.strip()
    
    def extract_action_from_response(
        self,
        response: str,
        action_names: List[str],
        candidate_indices: Optional[List[int]] = None
    ) -> Tuple[int, str]:
        """
        Extract the predicted action from the generated response.
        
        Args:
            response: Generated text from the model
            action_names: List of all action names
            candidate_indices: If provided, only match among these candidates
        
        Returns:
            (predicted_index, matched_action_name)
        """
        response_lower = response.lower()
        
        # Determine which actions to search
        if candidate_indices is not None:
            search_actions = [(idx, action_names[idx]) for idx in candidate_indices]
        else:
            search_actions = list(enumerate(action_names))
        
        # Find best match
        best_idx = search_actions[0][0]  # Default to first candidate
        best_score = 0
        best_action = search_actions[0][1]
        
        for idx, action in search_actions:
            action_clean = action.lower().replace("_", " ")
            
            # Exact match
            if action_clean in response_lower or action.lower() in response_lower:
                score = len(action_clean) * 2  # Prefer longer matches
                if score > best_score:
                    best_score = score
                    best_idx = idx
                    best_action = action
        
        return best_idx, best_action
    

#%% ========== Create wrapper class for inference ==========
class InferenceModel:
    """Wrapper to match LLMActionRerankerGenerative interface"""
    def __init__(self, peft_model, tokenizer, config):
        self.model = peft_model
        self.tokenizer = tokenizer
        self.config = config
    
    def eval(self):
        self.model.eval()
        return self
    
    def train(self):
        self.model.train()
        return self
    
    def parameters(self):
        return self.model.parameters()
    
    def __call__(self, **kwargs):
        return self.model(**kwargs)
    
    @torch.no_grad()
    def generate_with_reasoning(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int = 150
    ) -> str:
        """
        Generate action prediction WITH reasoning explanation.
        
        This is where the neuro-symbolic magic happens - the LLM explains
        which concepts led to its decision.
        """
        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Greedy decoding for reproducibility
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )
        
        # Decode only the generated part (not the prompt)
        generated_tokens = outputs[0, input_ids.shape[1]:]
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        return generated_text.strip()
    
    def extract_action_from_response(
        self,
        response: str,
        action_names: List[str],
        candidate_indices: Optional[List[int]] = None
    ) -> Tuple[int, str]:
        """
        Extract the predicted action from the generated response.
        
        Args:
            response: Generated text from the model
            action_names: List of all action names
            candidate_indices: If provided, only match among these candidates
        
        Returns:
            (predicted_index, matched_action_name)
        """
        response_lower = response.lower()
        
        # Determine which actions to search
        if candidate_indices is not None:
            search_actions = [(idx, action_names[idx]) for idx in candidate_indices]
        else:
            search_actions = list(enumerate(action_names))
        
        # Find best match
        best_idx = search_actions[0][0]  # Default to first candidate
        best_score = 0
        best_action = search_actions[0][1]
        
        for idx, action in search_actions:
            action_clean = action.lower().replace("_", " ")
            
            # Exact match
            if action_clean in response_lower or action.lower() in response_lower:
                score = len(action_clean) * 2  # Prefer longer matches
                if score > best_score:
                    best_score = score
                    best_idx = idx
                    best_action = action
        
        return best_idx, best_action
    














#%% Add class method

# class LLMActionRerankerGenerative(nn.Module):
#     # ...existing __init__ and other methods...
    
#     @classmethod
#     def from_checkpoint(cls, checkpoint_dir: str, config: TrainingConfig = None):
#         """
#         Load a trained model from checkpoint for inference.
        
#         Args:
#             checkpoint_dir: Path to saved checkpoint (with adapter_model.safetensors)
#             config: TrainingConfig (if None, loads from checkpoint)
        
#         Returns:
#             Loaded LLMActionRerankerGenerative ready for inference
#         """
#         from chkpt_utils import load_lora_model
#         import json
#         from pathlib import Path
        
#         checkpoint_path = Path(checkpoint_dir)
        
#         # Load config if not provided
#         if config is None:
#             config_path = checkpoint_path / "training_config.json"
#             if config_path.exists():
#                 with open(config_path, 'r') as f:
#                     config_dict = json.load(f)
#                 config = TrainingConfig(**config_dict)
#             else:
#                 config = TrainingConfig()
        
#         # Create instance (but we'll replace the model)
#         instance = cls.__new__(cls)
#         nn.Module.__init__(instance)
#         instance.config = config
        
#         # Load tokenizer
#         instance.tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
#         if instance.tokenizer.pad_token is None:
#             instance.tokenizer.pad_token = instance.tokenizer.eos_token
        
#         # Load model using our utility
#         peft_model, _, _ = load_lora_model(
#             checkpoint_dir=checkpoint_dir,
#             base_model_name=config.model_name,
#             device_map="auto",
#             use_4bit=config.use_4bit,
#             torch_dtype=torch.bfloat16
#         )
        
#         instance.model = peft_model
#         instance.eval()
        
#         print(f"✓ Loaded model from {checkpoint_dir}")
#         return instance


# Simple one-liner to load!
# model = LLMActionRerankerGenerative.from_checkpoint("./checkpoints/best_model")
# device = next(model.parameters()).device