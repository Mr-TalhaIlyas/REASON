from typing import List, Dict, Iterable
import numpy as np
import random
import pandas as pd


def build_slices(counts):
    """Return dict of slice objects per concept based on sequential layout."""
    slices = {}
    start = 0
    for k, n in counts.items():
        end = start + n
        slices[k] = slice(start, end)
        start = end
    return slices

def split_vector(vec, counts):
    """
    vec: 1D numpy array of length 80
    returns: dict with keys: parts (each includes interaction), 'full_body', 'indices'
    """
    concep_vec_dict = {}
    if vec.ndim != 1 or vec.shape[0] != sum(counts.values()):
        raise ValueError(f"Expected vec shape ({sum(counts.values())},), got {vec.shape}")

    sl = build_slices(counts)

    # Build individual parts (each + iteraction)
    interaction = vec[sl['interaction']]
    for part in ['head', 'hand', 'arm', 'hip', 'leg', 'foot']:
        # concep_vec_dict[part] = vec[sl[part]] # to not attach interaction # to NOT attach interaction
        concep_vec_dict[part] = np.concatenate([vec[sl[part]], interaction]) # to attach interaction
        
    # Build full_body = concat of all non-temporal parts + iteraction (once)
    concep_vec_dict['full_body'] = np.concatenate([vec[sl[p]] for p in ['head','hand','arm','hip','leg','foot']] + [interaction])
    # full body without adding interaction.
    # concep_vec_dict['full_body'] = np.concatenate([vec[sl[p]] for p in ['head','hand','arm','hip','leg','foot']])
    
    concep_vec_dict['temporal'] = vec[sl['temporal']]
    # Also return the exact index ranges for transparency
    # concep_vec_dict['indices'] = {k: (slc.start, slc.stop) for k, slc in sl.items()} # uncomment when needed

    return concep_vec_dict


def generate_temporal_prompts(
    concepts: Iterable[str],
    subject: str = "a person",
    context: str = "in a video for action recognition",
    variants_per_concept: int = 4
) -> Dict[str, List[str]]:
    """
    Turn temporal concept tokens into multiple human-readable prompts for CLIP text embeddings.

    Args:
        concepts: iterable of concept strings like
            'temporal_direction_motion_upward', 'temporal_sequence_seq_hands_first', ...
        subject: short description of the actor, e.g., 'a person', 'the subject'
        context: brief tail phrase keeping prompts grounded in action recognition
        variants_per_concept: cap the number of variants returned per concept

    Returns:
        dict: {concept -> [prompt variants]}
    """

    # Template banks per micro-type
    DIR_TEMPLATES = {
        "upward": [
            f"{subject} is moving upward {context}.",
            f"The motion is directed upward {context}.",
            f"{subject}'s movement trends upward {context}.",
            f"Upward motion characterizes the action {context}.",
            f"The action includes an upward trajectory {context}.",
        ],
        "downward": [
            f"{subject} is moving downward {context}.",
            f"The motion is directed downward {context}.",
            f"{subject}'s movement trends downward {context}.",
            f"Downward motion characterizes the action {context}.",
            f"The action includes a downward trajectory {context}."
        ],
        "forward": [
            f"{subject} is moving forward {context}.",
            f"The motion is directed forward {context}.",
            f"{subject} advances forward {context}.",
            f"Forward motion characterizes the action {context}.",
            f"The action proceeds in a forward direction {context}."
        ],
        "backward": [
            f"{subject} is moving backward {context}.",
            f"The motion is directed backward {context}.",
            f"{subject} shifts backward {context}.",
            f"Backward motion characterizes the action {context}.",
            f"The action proceeds in a backward direction {context}."
        ],
        "converging": [
            f"{subject}'s limbs move toward each other (converging) {context}.",
            f"The motion is converging, parts come together {context}.",
            f"Movement narrows inward as segments converge {context}.",
            f"Converging motion characterizes the action {context}.",
            f"Body parts approach each other during the action {context}."
        ],
        "diverging": [
            f"{subject}'s limbs move apart (diverging) {context}.",
            f"The motion is diverging, parts separate {context}.",
            f"Movement widens outward as segments diverge {context}.",
            f"Diverging motion characterizes the action {context}.",
            f"Body parts move away from each other during the action {context}."
        ],
        "reversible": [
            f"The motion is reversible and can be undone {context}.",
            f"{subject} performs a reversible movement pattern {context}.",
            f"The action exhibits reversibility in its motion {context}.",
            f"A reversible motion sequence is present {context}.",
            f"The movement can proceed and then reverse along the same path {context}."
        ],
    }

    SEQ_TEMPLATES = {
        "hands_first": [
            f"The movement initiates with the hands {context}.",
            f"{subject} starts the action using the hands first {context}.",
            f"Hand motion triggers the sequence before other parts {context}.",
            f"The action begins with hand initiation {context}.",
            f"Hands lead the movement sequence {context}."
        ],
        "legs_first": [
            f"The movement initiates with the legs {context}.",
            f"{subject} starts the action using the legs first {context}.",
            f"Leg motion triggers the sequence before other parts {context}.",
            f"The action begins with leg initiation {context}.",
            f"Legs lead the movement sequence {context}."
        ],
        "body_first": [
            f"The movement initiates with the torso/body {context}.",
            f"{subject} starts the action with body movement first {context}.",
            f"Body motion triggers the sequence before limbs {context}.",
            f"The action begins with body initiation {context}.",
            f"The torso leads the movement sequence {context}."
        ],
        "simultaneous": [
            f"Multiple body parts move simultaneously {context}.",
            f"The action exhibits synchronous motion {context}.",
            f"Movements occur at the same time across parts {context}.",
            f"Simultaneous coordination characterizes the sequence {context}.",
            f"Body segments activate together in parallel {context}."
        ],
        "alternating": [
            f"Body parts move in an alternating pattern {context}.",
            f"The action exhibits alternation between sides or parts {context}.",
            f"Movements switch back and forth in sequence {context}.",
            f"Alternating coordination characterizes the sequence {context}.",
            f"The pattern alternates between segments {context}."
        ],
        # "cascading": [
        #     f"The motion flows in a cascading sequence across parts {context}.",
        #     f"{subject} exhibits a wave-like cascade of movements {context}.",
        #     f"A cascading activation spreads from one part to another {context}.",
        #     f"The action unfolds as a cascade through the body {context}.",
        #     f"Sequential propagation forms a cascading pattern {context}."
        # ],
    }

    PHASE_TEMPLATES = {
        "preparation": [
            f"The action shows a preparation phase before execution {context}.",
            f"{subject} prepares posture and alignment before moving {context}.",
            f"A preparatory setup is visible prior to the main motion {context}.",
            f"The movement includes a distinct preparation phase {context}.",
            f"Readiness cues appear before execution {context}."
        ],
        "execution": [
            f"The action emphasizes the execution phase {context}.",
            f"{subject} is in the main execution of the movement {context}.",
            f"The core motion is being carried out {context}.",
            f"Execution dominates the observed segment {context}.",
            f"The primary movement unfolds during execution {context}."
        ],
        "retraction": [
            f"The action includes a retraction/return phase {context}.",
            f"{subject} returns to the starting posture {context}.",
            f"Movement retracts back after execution {context}.",
            f"A return-to-neutral phase is visible {context}.",
            f"The motion concludes with retraction {context}."
        ],
        "hold": [
            f"{subject} holds a position during the action {context}.",
            f"A static hold phase is present {context}.",
            f"The motion includes a maintained posture {context}.",
            f"Holding the position is part of the sequence {context}.",
            f"A brief isometric hold occurs {context}."
        ],
        "transition": [
            f"The action transitions between sub-phases {context}.",
            f"{subject} shifts smoothly between positions {context}.",
            f"Transitional movement links phases together {context}.",
            f"Intermediate transitions are visible {context}.",
            f"The motion contains phase-to-phase transitions {context}."
        ],
        "cyclic": [
            f"The action exhibits a cyclic pattern {context}.",
            f"{subject} repeats the movement in cycles {context}.",
            f"Repetitive periodic motion is present {context}.",
            f"Cyclic repetition characterizes the action {context}.",
            f"The sequence loops through repeated cycles {context}."
        ],
    }

    DYN_TEMPLATES = {
        "speed_slow": [
            f"The movement is slow and controlled {context}.",
            f"{subject} performs the action at a slow speed {context}.",
            f"Low-velocity motion characterizes the action {context}.",
            f"The sequence proceeds slowly {context}.",
            f"The pace is deliberately slow {context}."
        ],
        "speed_fast": [
            f"The movement is fast and energetic {context}.",
            f"{subject} performs the action at a high speed {context}.",
            f"High-velocity motion characterizes the action {context}.",
            f"The sequence proceeds quickly {context}.",
            f"The pace is notably fast {context}."
        ],
        "speed_accelerating": [
            f"The movement is accelerating over time {context}.",
            f"{subject} increases speed during the action {context}.",
            f"Velocity ramps up as the sequence unfolds {context}.",
            f"The pace transitions from slower to faster {context}.",
            f"Acceleration characterizes the motion profile {context}."
        ],
        "speed_decelerating": [
            f"The movement is decelerating over time {context}.",
            f"{subject} decreases speed during the action {context}.",
            f"Velocity tapers off as the sequence unfolds {context}.",
            f"The pace transitions from faster to slower {context}.",
            f"Deceleration characterizes the motion profile {context}."
        ],
        "rhythm_regular": [
            f"The movement follows a regular rhythm {context}.",
            f"{subject} keeps a steady, periodic beat {context}.",
            f"Rhythmic regularity characterizes the action {context}.",
            f"The cadence is consistent and even {context}.",
            f"A uniform rhythm underlies the motion {context}."
        ],
        "rhythm_irregular": [
            f"The movement follows an irregular rhythm {context}.",
            f"{subject} shows uneven, non-uniform timing {context}.",
            f"Rhythmic variability characterizes the action {context}.",
            f"The cadence is inconsistent and uneven {context}.",
            f"A non-uniform rhythm underlies the motion {context}."
        ],
        "duration_brief": [
            f"The action is brief in duration {context}.",
            f"{subject} performs a short, transient movement {context}.",
            f"A quick, short-lived motion is observed {context}.",
            f"The sequence occurs over a brief interval {context}.",
            f"The duration is notably short {context}."
        ],
        "duration_sustained": [
            f"The action is sustained over a longer duration {context}.",
            f"{subject} maintains the movement for an extended time {context}.",
            f"A prolonged motion is observed {context}.",
            f"The sequence persists over an extended interval {context}.",
            f"The duration is notably long {context}."
        ],
    }

    out: Dict[str, List[str]] = {}

    for c in concepts:
        # Normalize & route
        if c.startswith("temporal_direction_motion_"):
            key = c.split("temporal_direction_motion_")[-1]
            bank = DIR_TEMPLATES.get(key, None)

        elif c.startswith("temporal_sequence_seq_"):
            key = c.split("temporal_sequence_seq_")[-1]
            bank = SEQ_TEMPLATES.get(key, None)

        elif c.startswith("temporal_phase_phase_"):
            key = c.split("temporal_phase_phase_")[-1]
            bank = PHASE_TEMPLATES.get(key, None)

        elif c.startswith("temporal_dynamics_speed_"):
            key = "speed_" + c.split("temporal_dynamics_speed_")[-1]
            bank = DYN_TEMPLATES.get(key, None)

        elif c.startswith("temporal_dynamics_rhythm_"):
            key = "rhythm_" + c.split("temporal_dynamics_rhythm_")[-1]
            bank = DYN_TEMPLATES.get(key, None)

        elif c.startswith("temporal_dynamics_duration_"):
            key = "duration_" + c.split("temporal_dynamics_duration_")[-1]
            bank = DYN_TEMPLATES.get(key, None)

        else:
            bank = None

        if not bank:
            # Fallback generic phrasing if an unexpected token appears
            out[c] = [
                f"{subject} performs a movement labeled '{c}' {context}.",
                f"The action exhibits the temporal attribute '{c}' {context}.",
                f"Temporal characteristic detected: {c} {context}.",
                f"{subject}'s motion reflects '{c}' {context}."
            ][:variants_per_concept]
        else:
            out[c] = bank[:variants_per_concept]

    return out


def flatten_prompt_dict(prompt_dict: Dict[str, List[str]]) -> List[str]:
    """Flatten {concept: [prompts]} -> single list of prompts."""
    merged = []
    for _, v in prompt_dict.items():
        merged.extend(v)
    return merged

PROMPT_TEMPLATES2 = [
    lambda concept: f"a person's {concept}",
    lambda concept: f"{concept} movement",
    lambda concept: f"human action where {concept}",
    lambda concept: f"skeleton action: {concept}",
    lambda concept: f"action of performing {concept}",
    lambda concept: f"an action with {concept}",
    lambda concept: f"human skeleton action where {concept}",
]

def split_prompts_by_part(concept_df, ordered_concept_dict):
   
    concepts = concept_df.columns.tolist()[1:]

    parts = ["head", "hand", "arm", "hip", "leg", "foot"]

    concepts_by_part = {}
    for bp in parts:
        concepts_by_part[bp] = []

    for i, concept_info in enumerate(concepts):
        
        bps, movments = concept_info.split('_',  1)
        if bps != 'interaction' and bps != 'temporal':
            movments = ' '.join(movments.split('_'))
            
            template = random.choice(PROMPT_TEMPLATES2)
            
            prompt = template(movments)
            
            concepts_by_part[bps].append(prompt)
            
    combined_list = []
    for key in concepts_by_part.keys():  # Sort by keys to maintain a consistent order
        combined_list.extend(concepts_by_part[key])

    concepts_by_part['full_body'] = combined_list

    # for all body parts
    for key in concepts_by_part.keys():
        concepts_by_part[key].append('This action involves interaction of two persons.')
    # for full body only
    # combined_list['full_body'].append('This action involves interaction of two persons.')

    tmp_strt_idx = sum(list(ordered_concept_dict.values())[:-1])
    tmp_prompts = generate_temporal_prompts(concepts[tmp_strt_idx:], subject="a person",
                                        context="in a short action video for action recognition",
                                        variants_per_concept=1)# <- keep varients fixed here
    x = list(tmp_prompts.values())
    x = [x[i][0] for i in range(len(x))]
    concepts_by_part['temporal'] = x
    
    return concepts_by_part