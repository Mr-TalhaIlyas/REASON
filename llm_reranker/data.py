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

from config import TrainingConfig


class ConceptFormatter:
    """Formats concepts into structured text for LLM input"""
    
    def __init__(self, concept_df_path: str):
        self.concept_df = pd.read_csv(concept_df_path)
        self.action_names = self.concept_df["action_class"].values.tolist()
        self.all_concepts = self.concept_df.columns.tolist()[1:]
        # pop [50, 65] null concepts
        null_concept_index = [50, 65] # indices of 'null_concept' in vocab and concept_matrix
        self.all_concepts.pop(null_concept_index[1])
        self.all_concepts.pop(null_concept_index[0])
        
        # Create prototype concept vectors for each action
        self.action_prototypes = {}
        for _, row in self.concept_df.iterrows():
            action = row["action_class"]
            concepts = {col: row[col] for col in self.all_concepts}
            self.action_prototypes[action] = concepts
        
        # Organize concepts by category for better readability
        self.concept_categories = self._categorize_concepts()
    
    def _categorize_concepts(self) -> Dict[str, List[str]]:
        """Group concepts by body part and temporal category"""
        categories = {
            "head": [], "hand": [], "arm": [], "hip": [], 
            "leg": [], "foot": [], "interaction": [], "temporal": []
        }
        for concept in self.all_concepts:
            for cat in categories:
                if concept.startswith(cat) or concept.startswith("temporal"):
                    if concept.startswith("temporal"):
                        categories["temporal"].append(concept)
                    else:
                        categories[cat].append(concept)
                    break
        return categories
    
    def format_predicted_concepts(
        self, 
        concept_probs: np.ndarray, 
        threshold: float = 0.2
    ) -> str:
        """Convert predicted concept probabilities to readable text"""
        lines = []
        
        for category, concepts in self.concept_categories.items():
            category_concepts = []
            for i, concept in enumerate(self.all_concepts):
                if concept in concepts:
                    idx = self.all_concepts.index(concept)
                    prob = concept_probs[idx]
                    if prob >= threshold:
                        # Clean concept name and add confidence
                        clean_name = concept.replace("_", " ").replace(f"{category} ", "")
                        category_concepts.append(f"{clean_name} ({prob:.2f})")
            
            if category_concepts:
                lines.append(f"  {category.upper()}: {', '.join(category_concepts)}")
        
        return "\n".join(lines) if lines else "  No strong concepts detected"
    
    def format_candidate_actions(
        self,
        action_probs: np.ndarray,
        top_k: int = 5,
        is_train: bool = False
    ) -> Tuple[str, List[int]]:
        """Format top-k action candidates with their probabilities"""
        top_indices = np.argsort(action_probs)[::-1][:top_k].copy()
        if is_train: # shuffle top indices to avoid any ordering bias during training
            np.random.shuffle(top_indices)
        lines = []
        for rank, idx in enumerate(top_indices, 1):
            action = self.action_names[idx]
            if is_train:
                if np.random.rand() < 0.95: # 95% chance to mask prob during training
                    prob = 0.5 # randomly set to constant to avoid leaking info
                else:
                    prob = action_probs[idx] 
            else:
                prob = action_probs[idx]
            lines.append(f"  {rank}. {action} (confidence: {prob:.3f})")
        
        return "\n".join(lines), top_indices.tolist()
    
    def get_prototype_description(self, action_name: str) -> str:
        """Get the concept prototype for an action class"""
        if action_name not in self.action_prototypes:
            return ""
        
        prototype = self.action_prototypes[action_name]
        active_concepts = [c.replace("_", " ") for c, v in prototype.items() if v == 1]
        return ", ".join(active_concepts[:15])  # Limit for context length


class RegularizedGenerativeDataset(Dataset):
    """
    Dataset with optional augmentation for regularization.
    """
    
    def __init__(
        self,
        concept_predictions: np.ndarray,
        action_predictions: np.ndarray,
        true_labels: np.ndarray,
        formatter,
        tokenizer,
        config: TrainingConfig,
        is_train: bool = True,
        generate_reasoning: bool = True
    ):
        self.concept_preds = concept_predictions
        self.action_preds = action_predictions
        self.true_labels = true_labels
        self.formatter = formatter
        self.tokenizer = tokenizer
        self.config = config
        self.is_train = is_train
        self.generate_reasoning = generate_reasoning
        
    def __len__(self):
        return len(self.true_labels)
    
    def __getitem__(self, idx):
        concept_probs = self.concept_preds[idx].copy()  # Copy for augmentation
        action_probs = self.action_preds[idx].copy()
        true_label = self.true_labels[idx]
        
        # Apply augmentation (only during training)
        if self.is_train:
            concept_probs = self._augment_concepts(concept_probs)
        
        # Get target action name
        target_action = self.formatter.action_names[true_label]
        
        # Create prompt
        prompt = self._create_prompt(concept_probs, action_probs)
        
        # Create response
        if self.generate_reasoning:
            response = self._create_reasoning_response(target_action, concept_probs, true_label)
        else:
            response = target_action
        
        # Combine and tokenize
        full_text = f"{prompt}\n\nAnswer: {response}"
        
        full_encoding = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.config.max_seq_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        prompt_encoding = self.tokenizer(
            prompt + "\n\nAnswer:",
            truncation=True,
            max_length=self.config.max_seq_length,
        )
        prompt_length = len(prompt_encoding["input_ids"])
        
        labels = full_encoding["input_ids"].clone().squeeze(0)
        # 1. Mask the prompt
        labels[:prompt_length] = -100
        
        # 2. Mask the padding (CRITICAL FIX)
        # attention_mask is 0 for padding tokens
        attention_mask = full_encoding["attention_mask"].squeeze(0)
        labels[attention_mask == 0] = -100
        
        return {
            "input_ids": full_encoding["input_ids"].squeeze(0),
            "attention_mask": full_encoding["attention_mask"].squeeze(0),
            "labels": labels,
            "true_label": torch.tensor(true_label, dtype=torch.long),
            "prompt_length": prompt_length,
        }
    
    def _augment_concepts(self, concept_probs: np.ndarray) -> np.ndarray:
        """Apply data augmentation to concept probabilities"""
        
        # Add Gaussian noise
        if self.config.concept_noise_std > 0:
            noise = np.random.normal(0, self.config.concept_noise_std, concept_probs.shape)
            concept_probs = concept_probs + noise
            concept_probs = np.clip(concept_probs, 0, 1)
        
        # Random dropout of concepts
        if self.config.concept_dropout > 0:
            mask = np.random.random(concept_probs.shape) > self.config.concept_dropout
            concept_probs = concept_probs * mask
        
        return concept_probs
    
    def _create_prompt(self, concept_probs: np.ndarray, action_probs: np.ndarray) -> str:
        concept_text = self.formatter.format_predicted_concepts(concept_probs, threshold=self.config.concept_threshold)
        action_text, _ = self.formatter.format_candidate_actions(action_probs, top_k=self.config.top_k_actions, is_train=self.is_train)
        
        return f"""You are an expert action recognition system. Based on the detected body movement concepts from skeleton data, determine which action is being performed.

DETECTED MOVEMENT CONCEPTS (confidence scores):
{concept_text}

CANDIDATE ACTIONS (from vision model):
{action_text}

Which action is most likely? Explain your reasoning based on the concepts."""

    def _create_reasoning_response(self, target_action: str, concept_probs: np.ndarray, true_label: int) -> str:
        prototype = self.formatter.action_prototypes[target_action]
        
        matching_concepts = []
        for i, concept in enumerate(self.formatter.all_concepts):
            if prototype[concept] == 1 and concept_probs[i] >= 0.3:
                clean_name = concept.replace("_", " ")
                matching_concepts.append(f"{clean_name} ({concept_probs[i]:.2f})")
        
        if matching_concepts:
            reasoning = f"The action is {target_action}.\n\nKey supporting concepts:\n"
            reasoning += "\n".join(f"- {c}" for c in matching_concepts[:5])
        else:
            reasoning = f"The action is {target_action}."
        
        return reasoning
    
    
def create_mixed_dataset(
    train_concepts: np.ndarray,
    train_actions: np.ndarray,
    train_labels: np.ndarray,
    val_concepts: np.ndarray,
    val_actions: np.ndarray,
    val_labels: np.ndarray,
    mix_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Mix some validation samples into training set.
    
    This helps when:
    1. Val set has different distribution than train
    2. Model overfits to train-specific patterns
    
    IMPORTANT: Keep a separate held-out TEST set for final evaluation!
    
    Args:
        mix_ratio: Fraction of val samples to add to train (e.g., 0.1 = 10%)
    
    Returns:
        Mixed training data (concepts, actions, labels)
    """
    np.random.seed(42)
    
    n_val = len(val_labels)
    n_to_mix = int(n_val * mix_ratio)
    
    # Randomly select val samples to mix
    mix_indices = np.random.choice(n_val, n_to_mix, replace=False)
    
    # Concatenate
    mixed_concepts = np.concatenate([train_concepts, val_concepts[mix_indices]], axis=0)
    mixed_actions = np.concatenate([train_actions, val_actions[mix_indices]], axis=0)
    mixed_labels = np.concatenate([train_labels, val_labels[mix_indices]], axis=0)
    
    print(f"\n📊 Dataset mixing:")
    print(f"  Original train size: {len(train_labels)}")
    print(f"  Val samples added:   {n_to_mix} ({mix_ratio*100:.0f}% of val)")
    print(f"  New train size:      {len(mixed_labels)}")
    
    return mixed_concepts, mixed_actions, mixed_labels

def analyze_distribution_shift(
    train_concepts: np.ndarray,
    val_concepts: np.ndarray,
    concept_names: List[str]
):
    """
    Analyze if there's a distribution shift between train and val concepts.
    This helps diagnose why val accuracy might be lower.
    """
    print("\n" + "="*70)
    print("DISTRIBUTION SHIFT ANALYSIS")
    print("="*70)
    
    # Compare mean concept activations
    train_means = train_concepts.mean(axis=0)
    val_means = val_concepts.mean(axis=0)
    
    # Find concepts with biggest shift
    shifts = np.abs(train_means - val_means)
    top_shifts = np.argsort(shifts)[::-1][:10]
    
    print("\nTop 10 concepts with biggest distribution shift:")
    print("-" * 60)
    print(f"{'Concept':<40} {'Train Mean':>10} {'Val Mean':>10} {'Shift':>8}")
    print("-" * 60)
    
    for idx in top_shifts:
        concept = concept_names[idx] if idx < len(concept_names) else f"concept_{idx}"
        print(f"{concept:<40} {train_means[idx]:>10.3f} {val_means[idx]:>10.3f} {shifts[idx]:>8.3f}")
    
    # Overall statistics
    overall_shift = np.mean(shifts)
    print(f"\nOverall mean absolute shift: {overall_shift:.4f}")
    
    if overall_shift > 0.1:
        print("⚠️ Significant distribution shift detected!")
        print("   Consider: 1) Mixing val samples into train")
        print("            2) Domain adaptation techniques")
        print("            3) Checking if concept predictor generalizes well")
    else:
        print("✓ Distribution shift is relatively small")
    
    return shifts