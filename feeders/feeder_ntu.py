import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd
import json
import random

from feeders import tools
from feeders.prompt_tools import split_prompts_by_part, split_vector





def custom_collate_fn(batch):
    """
    Custom collate function that handles:
    1. Stacking tensors
    2. Batching prompts by body part
    3. Ready for CLIP tokenization on main process
    """
    data_numpy = torch.stack([torch.from_numpy(item[0]) for item in batch])
    label = torch.tensor([item[1] for item in batch])
    
    # Concept vectors grouped by body part - collate into dict of tensors
    concept_vectors_by_part = {}
    for key in batch[0][2].keys():
        concept_vectors_by_part[key] = torch.stack([
            torch.from_numpy(item[2][key]) for item in batch
        ])
    
    # Prompts by part - collate into dict of lists
    # Structure: {'head': [prompt1, prompt2, ...], 'hand': [...], ...}
    prompts_by_part = {}
    body_parts = ['head', 'hand', 'arm', 'hip', 'leg', 'foot', 'full_body', 'temporal']
    
    for bp in body_parts:
        # prompts_by_part[bp] = [item[3][bp] for item in batch]
        # only append once overall
        for item in batch:
            prompts_by_part[bp] = item[3][bp]
            break
    
    # action names
    # action_names = [item[4] for item in batch]
    
    return data_numpy, label, concept_vectors_by_part, prompts_by_part#, action_names


class Feeder(Dataset):
    def __init__(self, data_path, label_path=None, p_interval=1, split='train', random_choose=False, random_shift=False,
                 random_move=False, random_rot=False, window_size=-1, normalization=False, debug=False, use_mmap=False,
                 bone=False, vel=False, concepts_csv=None, remove_null_concept=True):
        """
        NTU Dataset Feeder with efficient prompt generation in workers.
        
        Args:
            data_path: Path to NTU skeleton data (.npz)
            concept_matrix_path: Path to concept matrix (.npy)
            vocabulary_path: Path to concept vocabulary (.json)
            concepts_csv: Path to concept CSV file
        """
        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.split = split
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
        self.use_mmap = use_mmap
        self.p_interval = p_interval
        self.random_rot = random_rot
        self.bone = bone
        self.vel = vel
        self.concepts_csv = concepts_csv
        self.remove_null_concept = remove_null_concept
        self.aug_prob_rot = 0.1
        self.aug_prob_move = 0.4
        self.load_data()
        if normalization:
            self.get_mean_map()

    def load_data(self):
        # Load skeleton data
        npz_data = np.load(self.data_path)
        if self.split == 'train':
            self.data = npz_data['x_train']
            self.label = np.where(npz_data['y_train'] > 0)[1]
            self.sample_name = ['train_' + str(i) for i in range(len(self.data))]
        elif self.split == 'test':
            self.data = npz_data['x_test']
            self.label = np.where(npz_data['y_test'] > 0)[1]
            self.sample_name = ['test_' + str(i) for i in range(len(self.data))]
        else:
            raise NotImplementedError('data split only supports train/test')
        
        N, T, _ = self.data.shape
        self.data = self.data.reshape((N, T, 2, 25, 3)).transpose(0, 4, 1, 3, 2)
        
        # Load concept matrix and vocabulary
        if self.concepts_csv is not None:
            self.concept_df = pd.read_csv(self.concepts_csv)
        else:
            raise ValueError("concepts_csv must be provided...")
        self.get_concepts_prelims()

    def get_concepts_prelims(self):
        """
        Get preliminary concept information for a specific label.
        """
        arr = self.concept_df.values
        arr = arr[:, 1:]
        self.concept_matrix_np = arr.astype(np.uint8)

        self.action_names = self.concept_df["action_class"].values

        self.vocab = self.concept_df.columns.tolist()[1:]
        if self.remove_null_concept:
            null_concept_index = [50, 65] # indices of 'null_concept' in vocab and concept_matrix
            self.vocab.pop(null_concept_index[1])
            self.vocab.pop(null_concept_index[0])
            self.concept_matrix_np = np.delete(self.concept_matrix_np, null_concept_index, axis=1)
            
        assert len(self.vocab) == self.concept_matrix_np.shape[1]

        self.concepts = []
        for i in range(len(self.vocab)):
            bp = self.vocab[i]
            self.concepts.append(self.vocab[i].split('_', 1)[0])

        x = np.asarray(self.concepts)
        x = np.unique(x, return_counts=True)
        y, z = x[0].tolist(), x[1].tolist()
        self.concept_counts = dict(zip(y, z))

        self.ordered_concept_dict = {key: self.concept_counts[key] for key in self.concepts}
        
        return self.concept_matrix_np, self.action_names, self.vocab,\
                self.concepts, self.concept_counts, self.ordered_concept_dict

    def get_mean_map(self):
        data = self.data
        N, C, T, V, M = data.shape
        self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))

    def __len__(self):
        return len(self.label)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        """
        Get a single sample with prompts generated in worker.
        
        Returns:
            data_numpy: Skeleton data (C, T, V, M)
            label: Action class label (int)
            concept_vectors_by_part: Dict of concept vectors grouped by body part
            prompts_by_part: Dict of prompt strings (NOT tokenized yet)
        """
        data_numpy = self.data[index]
        label = self.label[index]
        action_name = self.action_names[label]
        # Get concept vector for the label
        concept_vector = self.concept_matrix_np[label]
        # Split concept vector by body part
        concept_vectors_by_part = split_vector(concept_vector, self.ordered_concept_dict)
        # get prompts by part
        prompts_by_part = split_prompts_by_part(self.concept_df, self.ordered_concept_dict)
        
        # Process skeleton data
        data_numpy = np.array(data_numpy)
        valid_frame_num = np.sum(data_numpy.sum(0).sum(-1).sum(-1) != 0)
        data_numpy = tools.valid_crop_resize(data_numpy, valid_frame_num, self.p_interval, self.window_size)
        
        if self.split == 'train':
            aug_type = self._sample_augmentation()
            
            if aug_type == 'rotation' and self.random_rot:
                data_numpy = tools.random_rot(data_numpy, theta=0.3).numpy()
                # print("applied move augmentation")
            elif aug_type == 'move' and self.random_move:
                data_numpy = tools.random_move(data_numpy,
                                               angle_candidate=[-5., 5.],
                                               scale_candidate=[0.5, 0.9, 1.1, 1.5],
                                               transform_candidate=[-0.2, -0.1,
                                                                     0.1, 0.2,],
                                               move_time_candidate=[1])
                # print("applied move augmentation")
            # data_numpy = np.array(data_numpy)
            
        # if self.random_rot:
        #     data_numpy = tools.random_rot(data_numpy)
        if self.bone:
            from .bone_pairs import ntu_pairs
            bone_data_numpy = np.zeros_like(data_numpy)
            for v1, v2 in ntu_pairs:
                bone_data_numpy[:, :, v1 - 1] = data_numpy[:, :, v1 - 1] - data_numpy[:, :, v2 - 1]
            data_numpy = bone_data_numpy
        # elif not self.vel:  # Add centering for consistency
            # trajectory = data_numpy[:, :, 20]
            # data_numpy = data_numpy - data_numpy[:, :, 20:21]
            # data_numpy[:, :, 20] = trajectory
        if self.vel:
            data_numpy[:, :-1] = data_numpy[:, 1:] - data_numpy[:, :-1]
            data_numpy[:, -1] = 0

        return data_numpy, label, concept_vectors_by_part, prompts_by_part#, action_name

    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)


    def _sample_augmentation(self):
        """
        Sample augmentation type based on probabilities.
        
        Returns:
            str: 'rotation', 'move', or 'none'
        """
        rand_val = random.random()  # Random float in [0, 1)
        
        if rand_val < self.aug_prob_rot:
            # print("applying rotation augmentation")
            return 'rotation'
        elif rand_val < self.aug_prob_rot + self.aug_prob_move:
            # print("applying move augmentation")
            return 'move'
        else:
            # print("no augmentation applied")
            return 'none'




