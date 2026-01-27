#%%
"""
Data Sanity Check Script
========================
Visualizes the entire data pipeline:
1. Raw inputs (concepts, actions, labels)
2. Formatted prompts and responses
3. Tokenization details
4. Label masking for training
5. What tokens the model actually learns from
"""

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import List, Dict
import textwrap

from data import ConceptFormatter, RegularizedGenerativeDataset
from config import TrainingConfig


def print_separator(title: str, char: str = "=", width: int = 80):
    """Print a formatted separator"""
    print(f"\n{char * width}")
    print(f" {title}")
    print(f"{char * width}\n")


def visualize_raw_inputs(
    concept_probs: np.ndarray,
    action_probs: np.ndarray,
    true_label: int,
    formatter: ConceptFormatter,
    config: TrainingConfig,
    sample_idx: int = 0
):
    """Visualize raw numpy inputs before any processing"""
    
    print_separator(f"STEP 1: RAW INPUTS (Sample {sample_idx})")
    
    # Concept probabilities
    print("📊 CONCEPT PROBABILITIES:")
    print(f"   Shape: {concept_probs.shape}")
    print(f"   Min: {concept_probs.min():.4f}, Max: {concept_probs.max():.4f}, Mean: {concept_probs.mean():.4f}")
    
    # Top concepts
    top_concept_indices = np.argsort(concept_probs)[::-1][:10]
    print(f"\n   Top 10 Active Concepts (threshold={config.concept_threshold}):")
    for i, idx in enumerate(top_concept_indices):
        concept_name = formatter.all_concepts[idx] if idx < len(formatter.all_concepts) else f"concept_{idx}"
        prob = concept_probs[idx]
        marker = "✓" if prob >= config.concept_threshold else "✗"
        print(f"   {i+1:2d}. [{marker}] {concept_name:<40} = {prob:.4f}")
    
    # Action probabilities
    print(f"\n📊 ACTION PROBABILITIES:")
    print(f"   Shape: {action_probs.shape}")
    print(f"   Min: {action_probs.min():.4f}, Max: {action_probs.max():.4f}")
    
    top_action_indices = np.argsort(action_probs)[::-1][:10]
    print(f"\n   Top 10 Predicted Actions:")
    for i, idx in enumerate(top_action_indices):
        action_name = formatter.action_names[idx]
        prob = action_probs[idx]
        marker = "★" if idx == true_label else " "
        print(f"   {i+1:2d}. {marker} {action_name:<40} = {prob:.4f}")
    
    # True label
    true_action = formatter.action_names[true_label]
    true_action_rank = np.where(top_action_indices == true_label)[0]
    rank_str = f"(rank {true_action_rank[0]+1})" if len(true_action_rank) > 0 else "(not in top 10)"
    
    print(f"\n🎯 TRUE LABEL:")
    print(f"   Index: {true_label}")
    print(f"   Action: {true_action}")
    print(f"   Position in predictions: {rank_str}")
    
    return true_action


def visualize_formatted_text(
    concept_probs: np.ndarray,
    action_probs: np.ndarray,
    true_label: int,
    formatter: ConceptFormatter,
    config: TrainingConfig
):
    """Visualize the formatted prompt and response"""
    
    print_separator("STEP 2: FORMATTED TEXT (Before Tokenization)")
    
    # Format concepts
    concept_text = formatter.format_predicted_concepts(concept_probs, threshold=config.concept_threshold)
    print("📝 FORMATTED CONCEPTS:")
    print("-" * 60)
    print(concept_text)
    print("-" * 60)
    
    # Format candidate actions
    action_text, candidate_indices = formatter.format_candidate_actions(
        action_probs, top_k=config.top_k_actions
    )
    print(f"\n📝 CANDIDATE ACTIONS (top_k={config.top_k_actions}):")
    print(f"   Candidate indices: {candidate_indices}")
    print(f"   True label {true_label} in candidates: {true_label in candidate_indices}")
    print("-" * 60)
    print(action_text)
    print("-" * 60)
    
    # Create full prompt
    prompt = f"""You are an expert action recognition system. Based on the detected body movement concepts from skeleton data, determine which action is being performed.

DETECTED MOVEMENT CONCEPTS (confidence scores):
{concept_text}

CANDIDATE ACTIONS (from vision model):
{action_text}

Which action is most likely? Explain your reasoning based on the concepts."""

    print(f"\n📝 FULL PROMPT:")
    print("=" * 60)
    print(prompt)
    print("=" * 60)
    print(f"\n   Prompt length (characters): {len(prompt)}")
    
    # Create response
    target_action = formatter.action_names[true_label]
    prototype = formatter.action_prototypes[target_action]
    
    matching_concepts = []
    for i, concept in enumerate(formatter.all_concepts):
        if prototype[concept] == 1 and concept_probs[i] >= 0.3:
            clean_name = concept.replace("_", " ")
            matching_concepts.append(f"{clean_name} ({concept_probs[i]:.2f})")
    
    if matching_concepts:
        response = f"The action is {target_action}.\n\nKey supporting concepts:\n"
        response += "\n".join(f"- {c}" for c in matching_concepts[:5])
    else:
        response = f"The action is {target_action}."
    
    print(f"\n📝 TARGET RESPONSE:")
    print("=" * 60)
    print(response)
    print("=" * 60)
    print(f"\n   Response length (characters): {len(response)}")
    
    # Full text
    full_text = f"{prompt}\n\nAnswer: {response}"
    print(f"\n📝 FULL TEXT (Prompt + Response):")
    print(f"   Total length (characters): {len(full_text)}")
    
    return prompt, response, full_text


def visualize_tokenization(
    prompt: str,
    response: str,
    full_text: str,
    tokenizer,
    config: TrainingConfig
):
    """Visualize tokenization details"""
    
    print_separator("STEP 3: TOKENIZATION")
    
    # Tokenize prompt only
    prompt_with_answer = prompt + "\n\nAnswer:"
    prompt_encoding = tokenizer(
        prompt_with_answer,
        truncation=True,
        max_length=config.max_seq_length,
    )
    prompt_length = len(prompt_encoding["input_ids"])
    
    print(f"📊 PROMPT TOKENIZATION:")
    print(f"   Text: '{prompt_with_answer[-50:]}...' (last 50 chars)")
    print(f"   Token count: {prompt_length}")
    
    # Tokenize full text
    full_encoding = tokenizer(
        full_text,
        truncation=True,
        max_length=config.max_seq_length,
        padding="max_length",
        return_tensors="pt"
    )
    
    total_tokens = (full_encoding["attention_mask"].squeeze() == 1).sum().item()
    padding_tokens = (full_encoding["attention_mask"].squeeze() == 0).sum().item()
    response_tokens = total_tokens - prompt_length
    
    print(f"\n📊 FULL TEXT TOKENIZATION:")
    print(f"   Max sequence length: {config.max_seq_length}")
    print(f"   Total tokens (non-padding): {total_tokens}")
    print(f"   Padding tokens: {padding_tokens}")
    print(f"   Prompt tokens: {prompt_length}")
    print(f"   Response tokens: {response_tokens}")
    
    # Show actual tokens
    input_ids = full_encoding["input_ids"].squeeze()
    
    print(f"\n📝 TOKEN BREAKDOWN:")
    print("-" * 80)
    
    # Prompt tokens (first 20 and last 20)
    print("   PROMPT TOKENS (first 15):")
    for i in range(min(15, prompt_length)):
        token_id = input_ids[i].item()
        token_text = tokenizer.decode([token_id])
        print(f"      [{i:4d}] ID={token_id:6d} → '{repr(token_text)}'")
    
    if prompt_length > 30:
        print(f"      ... ({prompt_length - 30} tokens omitted) ...")
    
    print(f"\n   PROMPT TOKENS (last 15, before 'Answer:'):")
    for i in range(max(0, prompt_length - 15), prompt_length):
        token_id = input_ids[i].item()
        token_text = tokenizer.decode([token_id])
        print(f"      [{i:4d}] ID={token_id:6d} → '{repr(token_text)}'")
    
    # Response tokens
    print(f"\n   RESPONSE TOKENS (what model learns to generate):")
    for i in range(prompt_length, min(prompt_length + 30, total_tokens)):
        token_id = input_ids[i].item()
        token_text = tokenizer.decode([token_id])
        print(f"      [{i:4d}] ID={token_id:6d} → '{repr(token_text)}'")
    
    if response_tokens > 30:
        print(f"      ... ({response_tokens - 30} more response tokens) ...")
    
    # Padding tokens
    if padding_tokens > 0:
        print(f"\n   PADDING TOKENS (ignored in loss):")
        pad_start = total_tokens
        print(f"      [{pad_start}:{config.max_seq_length}] = {padding_tokens} × PAD (ID={tokenizer.pad_token_id})")
    
    return full_encoding, prompt_length


def visualize_labels_and_masking(
    full_encoding: Dict,
    prompt_length: int,
    tokenizer,
    config: TrainingConfig
):
    """Visualize how labels are masked for training"""
    
    print_separator("STEP 4: LABEL MASKING (What Model Actually Learns)")
    
    input_ids = full_encoding["input_ids"].squeeze()
    attention_mask = full_encoding["attention_mask"].squeeze()
    
    # Create labels as done in dataset
    labels = input_ids.clone()
    
    # Mask prompt
    labels[:prompt_length] = -100
    
    # Mask padding
    labels[attention_mask == 0] = -100
    
    # Count different types
    total_positions = len(labels)
    prompt_masked = prompt_length
    padding_masked = (attention_mask == 0).sum().item()
    trainable_positions = (labels != -100).sum().item()
    
    print(f"📊 LABEL STATISTICS:")
    print(f"   Total positions: {total_positions}")
    print(f"   Prompt positions (masked with -100): {prompt_masked}")
    print(f"   Padding positions (masked with -100): {padding_masked}")
    print(f"   Trainable positions (actual loss computed): {trainable_positions}")
    print(f"\n   ⚠️  Model ONLY learns from {trainable_positions} tokens!")
    
    # Visualize the masking
    print(f"\n📝 LABEL VISUALIZATION:")
    print("-" * 80)
    
    # Show transition from masked to unmasked
    transition_start = max(0, prompt_length - 5)
    transition_end = min(len(labels), prompt_length + 15)
    
    print(f"   Around prompt/response boundary (positions {transition_start}-{transition_end}):")
    print(f"   {'Pos':>6} {'InputID':>8} {'Label':>8} {'Token':>20} {'Status':<15}")
    print("   " + "-" * 70)
    
    for i in range(transition_start, transition_end):
        token_id = input_ids[i].item()
        label_id = labels[i].item()
        token_text = tokenizer.decode([token_id])[:15]
        
        if i < prompt_length:
            status = "🚫 PROMPT (masked)"
        elif attention_mask[i] == 0:
            status = "🚫 PADDING (masked)"
        else:
            status = "✅ TRAINABLE"
        
        label_str = str(label_id) if label_id != -100 else "-100"
        print(f"   {i:>6} {token_id:>8} {label_str:>8} {repr(token_text):>20} {status:<15}")
    
    # Show some trainable tokens
    trainable_indices = (labels != -100).nonzero().squeeze()
    if trainable_indices.numel() > 0:
        print(f"\n   First 10 TRAINABLE positions (model learns these):")
        for i, idx in enumerate(trainable_indices[:10].tolist()):
            token_id = input_ids[idx].item()
            label_id = labels[idx].item()
            token_text = tokenizer.decode([token_id])
            print(f"      [{idx:4d}] Label={label_id:6d} → '{repr(token_text)}'")
    
    return labels


def visualize_batch_statistics(dataloader: DataLoader, tokenizer, num_batches: int = 3):
    """Visualize statistics across multiple batches"""
    
    print_separator("STEP 5: BATCH STATISTICS")
    
    all_prompt_lengths = []
    all_trainable_counts = []
    all_total_tokens = []
    
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= num_batches:
            break
        
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        prompt_lengths = batch["prompt_length"]
        true_labels = batch["true_label"]
        
        batch_size = input_ids.shape[0]
        
        print(f"📦 BATCH {batch_idx + 1}:")
        print(f"   Batch size: {batch_size}")
        print(f"   Input shape: {input_ids.shape}")
        
        for i in range(min(3, batch_size)):  # Show first 3 samples
            prompt_len = prompt_lengths[i].item()
            trainable = (labels[i] != -100).sum().item()
            total = (attention_mask[i] == 1).sum().item()
            true_label = true_labels[i].item()
            
            all_prompt_lengths.append(prompt_len)
            all_trainable_counts.append(trainable)
            all_total_tokens.append(total)
            
            print(f"\n   Sample {i}:")
            print(f"      True label index: {true_label}")
            print(f"      Total tokens: {total}")
            print(f"      Prompt length: {prompt_len}")
            print(f"      Trainable tokens: {trainable}")
            print(f"      Loss computed on: {trainable/total*100:.1f}% of tokens")
            
            # Decode first few trainable tokens
            trainable_mask = labels[i] != -100
            if trainable_mask.any():
                trainable_ids = input_ids[i][trainable_mask][:10]
                trainable_text = tokenizer.decode(trainable_ids)
                print(f"      First trainable tokens: '{trainable_text[:50]}...'")
    
    # Summary statistics
    if all_prompt_lengths:
        print(f"\n📊 SUMMARY ACROSS {len(all_prompt_lengths)} SAMPLES:")
        print(f"   Prompt length: min={min(all_prompt_lengths)}, max={max(all_prompt_lengths)}, avg={np.mean(all_prompt_lengths):.1f}")
        print(f"   Trainable tokens: min={min(all_trainable_counts)}, max={max(all_trainable_counts)}, avg={np.mean(all_trainable_counts):.1f}")
        print(f"   Total tokens: min={min(all_total_tokens)}, max={max(all_total_tokens)}, avg={np.mean(all_total_tokens):.1f}")


def visualize_model_forward_pass(
    batch: Dict,
    model,
    tokenizer
):
    """Visualize what happens during a forward pass"""
    
    print_separator("STEP 6: MODEL FORWARD PASS")
    
    device = next(model.parameters()).device
    
    input_ids = batch["input_ids"][:1].to(device)  # Single sample
    attention_mask = batch["attention_mask"][:1].to(device)
    labels = batch["labels"][:1].to(device)
    
    print(f"📊 INPUT TO MODEL:")
    print(f"   input_ids shape: {input_ids.shape}")
    print(f"   attention_mask shape: {attention_mask.shape}")
    print(f"   labels shape: {labels.shape}")
    
    # Forward pass
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
    
    print(f"\n📊 MODEL OUTPUT:")
    print(f"   Loss: {outputs.loss.item():.4f}")
    print(f"   Logits shape: {outputs.logits.shape}")
    
    # Analyze predictions on trainable positions
    logits = outputs.logits.squeeze()  # [L, vocab_size]
    trainable_mask = labels.squeeze() != -100
    
    if trainable_mask.any():
        trainable_indices = trainable_mask.nonzero().squeeze()
        
        print(f"\n📝 PREDICTIONS ON TRAINABLE POSITIONS (first 5):")
        print(f"   {'Pos':>5} {'True Token':>20} {'Pred Token':>20} {'Correct':>8}")
        print("   " + "-" * 60)
        
        correct = 0
        total = 0
        
        for idx in trainable_indices[:5].tolist() if trainable_indices.dim() > 0 else [trainable_indices.item()]:
            # For position idx, the model predicts the NEXT token
            # But in causal LM, logits[idx] predicts token at position idx
            # The label at idx is the token the model should predict
            
            true_token_id = labels[0, idx].item()
            pred_token_id = logits[idx].argmax().item()
            
            true_token = tokenizer.decode([true_token_id])
            pred_token = tokenizer.decode([pred_token_id])
            
            is_correct = "✅" if true_token_id == pred_token_id else "❌"
            if true_token_id == pred_token_id:
                correct += 1
            total += 1
            
            print(f"   {idx:>5} {repr(true_token):>20} {repr(pred_token):>20} {is_correct:>8}")
        
        # Token-level accuracy (just for this sample)
        all_trainable = trainable_indices.tolist() if trainable_indices.dim() > 0 else [trainable_indices.item()]
        all_correct = sum(
            1 for idx in all_trainable 
            if labels[0, idx].item() == logits[idx].argmax().item()
        )
        print(f"\n   Token-level accuracy on this sample: {all_correct}/{len(all_trainable)} = {all_correct/len(all_trainable)*100:.1f}%")


def run_full_sanity_check(sample_indices: List[int] = [0, 100, 1000]):
    """Run complete sanity check on specified samples"""
    
    print("\n" + "🔍" * 40)
    print(" " * 30 + "DATA SANITY CHECK")
    print("🔍" * 40 + "\n")
    
    # Load config and data
    config = TrainingConfig()
    formatter = ConceptFormatter("/data_hdd/talha/nsai/scal/llm/data/ntu_rgbd_spatial_temporal.csv")
    
    print("Loading data...")
    train_concept_preds = np.load("/data_hdd/talha/nsai/scal/llm/data/train_concept_probs.npy")
    train_action_preds = np.load("/data_hdd/talha/nsai/scal/llm/data/train_action_probs.npy")
    train_labels = np.load("/data_hdd/talha/nsai/scal/llm/data/train_action_labels.npy")
    
    print(f"Dataset size: {len(train_labels)} samples")
    print(f"Concept dimensions: {train_concept_preds.shape}")
    print(f"Action dimensions: {train_action_preds.shape}")
    
    # Load tokenizer (without full model for speed)
    from transformers import AutoTokenizer
    print(f"\nLoading tokenizer: {config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Pad token: '{tokenizer.pad_token}' (ID: {tokenizer.pad_token_id})")
    print(f"EOS token: '{tokenizer.eos_token}' (ID: {tokenizer.eos_token_id})")
    
    # Analyze individual samples
    for sample_idx in sample_indices:
        if sample_idx >= len(train_labels):
            print(f"\n⚠️ Sample {sample_idx} out of range, skipping...")
            continue
        
        print("\n" + "=" * 80)
        print(f" ANALYZING SAMPLE {sample_idx}")
        print("=" * 80)
        
        concept_probs = train_concept_preds[sample_idx]
        action_probs = train_action_preds[sample_idx]
        true_label = train_labels[sample_idx]
        
        # Step 1: Raw inputs
        visualize_raw_inputs(concept_probs, action_probs, true_label, formatter, config, sample_idx)
        
        # Step 2: Formatted text
        prompt, response, full_text = visualize_formatted_text(
            concept_probs, action_probs, true_label, formatter, config
        )
        
        # Step 3: Tokenization
        full_encoding, prompt_length = visualize_tokenization(
            prompt, response, full_text, tokenizer, config
        )
        
        # Step 4: Label masking
        visualize_labels_and_masking(full_encoding, prompt_length, tokenizer, config)
    
    # Create dataset and check batches
    print("\n" + "=" * 80)
    print(" DATASET & DATALOADER CHECK")
    print("=" * 80)
    
    dataset = RegularizedGenerativeDataset(
        train_concept_preds,
        train_action_preds,
        train_labels,
        formatter,
        tokenizer,
        config,
        is_train=True
    )
    
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    
    visualize_batch_statistics(dataloader, tokenizer, num_batches=2)
    
    # Optional: Model forward pass (requires loading full model)
    print("\n" + "-" * 80)
    user_input = input("Do you want to test model forward pass? This loads the full model. (y/n): ")
    
    if user_input.lower() == 'y':
        print("\nLoading model...")
        from model import LLMActionRerankerGenerative
        model = LLMActionRerankerGenerative(config)
        
        # Get a batch
        batch = next(iter(dataloader))
        visualize_model_forward_pass(batch, model, tokenizer)
    
    print_separator("SANITY CHECK COMPLETE", char="✅", width=80)


def quick_check(sample_idx: int = 0):
    """Quick check of a single sample without loading the model"""
    run_full_sanity_check(sample_indices=[sample_idx])


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Data Sanity Check for LLM Training Pipeline")
    parser.add_argument("--samples", type=int, nargs="+", default=[0, 50, 100],
                       help="Sample indices to analyze")
    parser.add_argument("--quick", action="store_true",
                       help="Quick check of first sample only")
    
    args = parser.parse_args()
    
    if args.quick:
        quick_check(0)
    else:
        run_full_sanity_check(sample_indices=args.samples)
#%%