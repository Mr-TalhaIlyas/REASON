#%%
import os

os.chdir(os.path.dirname(__file__))
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import yaml
import time
from pathlib import Path
import random
import numpy as np
from tqdm import tqdm
import pandas as pd

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import torch.nn.functional as F

import wandb
from torchmetrics import Accuracy
from torchmetrics.classification import MultilabelExactMatch, MultilabelAccuracy
import sklearn.metrics as skm

import yaml
import torch
import numpy as np
import random
import wandb
from tqdm import tqdm
from pathlib import Path
import time

from feeders.feeder_ntu import Feeder, custom_collate_fn
# from model.model import SkeletonACL
from model.model_v8 import SkeletonACL_CLIP_Logic

from eval_utils import find_optimal_threshold, evaluate_multilabel

def init_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

with open('/data_hdd/talha/nsai/scal/config/config_skeleton_crl.yaml', 'r') as f:
    args = yaml.safe_load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SkeletonACL_CLIP_Logic(args, device).to(device)
#%% checkpoint_epoch_40 best_model 
chkpt = '/data_hdd/talha/nsai/scal/work_dir/v5exp7/best_model.pth'
model.load_state_dict(torch.load(chkpt)['model_state'])
model.eval()
print(f"Loaded model from {chkpt}")
#%%
data_loader = {}
data_loader['train'] = torch.utils.data.DataLoader(
    dataset=Feeder(**args['train_feeder_args']),
    batch_size=args['batch_size'],
    shuffle=False,
    num_workers=args['num_worker'],
    drop_last=True,
    collate_fn=custom_collate_fn,
    worker_init_fn=lambda _: init_seed(args.get('seed', 1)),
    pin_memory=True,
    persistent_workers=True if args['num_worker'] > 0 else False
)
data_loader['test'] = torch.utils.data.DataLoader(
    dataset=Feeder(**args['test_feeder_args']),
    batch_size=args['test_batch_size'],
    shuffle=False,
    num_workers=args['num_worker'],
    drop_last=True,
    collate_fn=custom_collate_fn,
    worker_init_fn=lambda _: init_seed(args.get('seed', 1)),
    pin_memory=True,
    persistent_workers=True if args['num_worker'] > 0 else False
)

concept_df = pd.read_csv('/data_hdd/talha/nsai/scal/feeders/ntu_rgbd_spatial_temporal.csv')
action_names = concept_df["action_class"].values
all_concepts = concept_df.columns.tolist()[1:]
#%%
THRESH = 0.5
epoch = 0
pbar = tqdm(
        enumerate(data_loader['test']),
        total=len(data_loader['test']),
        desc="[INFER]",
        # ncols=30
    )

total_accuracy = []
total_action_accuracy = []
for step, (batch_data, batch_label, batch_concept_vecs, batch_prompts) in pbar:
    batch_data = batch_data.to(device)
    batch_label = batch_label.to(device)
    s_trgt = batch_concept_vecs['full_body'].to(device)
    t_trgt = batch_concept_vecs['temporal'].to(device)
    
    with torch.cuda.amp.autocast(enabled=True):
        with torch.inference_mode():
            action_probs, _, _, op_dict = model(
                            batch_data, batch_label, batch_concept_vecs, batch_prompts, epoch
                        )
    concept_logits = op_dict['all_logits'] # all concept probabilites 
    concept_gts = torch.cat([s_trgt, t_trgt], dim=1)
    # print("Concept logits shape:", concept_logits.shape) # torch.Size([B, 80])
    # print("Concept GTs shape:", concept_gts.shape) # # torch.Size([B, 80])
    
    concept_probs = concept_logits # sigmoind is applied inside the model during training
    concept_preds = (concept_probs >= THRESH).float()
    
    batch_accuracy = skm.accuracy_score(concept_gts.cpu().numpy(),
                                        concept_preds.cpu().numpy())
    
    action_accuracy = skm.accuracy_score(batch_label.cpu().numpy(),
                                        torch.argmax(action_probs, dim=1).cpu().numpy())
    
    total_accuracy.append(batch_accuracy)
    total_action_accuracy.append(action_accuracy)

acc = np.mean(total_accuracy)
action_acc = np.mean(total_action_accuracy)
print(f"Action Prediction Accuracy: {action_acc*100:.2f}%")
print(f"Concept Prediction Accuracy: {acc*100:.2f}%")
# %%



def extract_predictions_from_model(
    model,  # Your SkeletonACL_CLIP_Logic model
    data_loader,
    device
):
    """
    Extract concept and action predictions from your skeleton model.
    This integrates with your existing eval loop.
    """
    all_concept_probs = []
    all_action_probs = []
    all_labels = []
    
    model.eval()
    for step, (batch_data, batch_label, batch_concept_vecs, batch_prompts) in tqdm(
        enumerate(data_loader), total=len(data_loader), desc="Extracting predictions"
    ):
        batch_data = batch_data.to(device)
        batch_label = batch_label.to(device)
        
        with torch.cuda.amp.autocast(enabled=True):
            with torch.inference_mode():
                action_probs, _, _, op_dict = model(
                    batch_data, batch_label, batch_concept_vecs, batch_prompts, epoch=0
                )
        
        concept_logits = op_dict['all_logits']
        
        # Store predictions
        all_concept_probs.append(torch.sigmoid(concept_logits).cpu().numpy())
        all_action_probs.append(action_probs.cpu().numpy())
        all_labels.append(batch_label.cpu().numpy())
    
    return (
        np.concatenate(all_concept_probs, axis=0),
        np.concatenate(all_action_probs, axis=0),
        np.concatenate(all_labels, axis=0)
    )
    
concep, action, labels = extract_predictions_from_model(
    model, data_loader['test'], device
)

np.save('/data_hdd/talha/nsai/scal/llm/data/test_concept_probs.npy', concep)
np.save('/data_hdd/talha/nsai/scal/llm/data/test_action_probs.npy', action)
np.save('/data_hdd/talha/nsai/scal/llm/data/test_action_labels.npy', labels)

concep, action, labels = extract_predictions_from_model(
    model, data_loader['train'], device
)

np.save('/data_hdd/talha/nsai/scal/llm/data/train_concept_probs.npy', concep)
np.save('/data_hdd/talha/nsai/scal/llm/data/train_action_probs.npy', action)
np.save('/data_hdd/talha/nsai/scal/llm/data/train_action_labels.npy', labels)
#%%

# let's calculate action accuracy on saved files

action_probs = np.load('/data_hdd/talha/nsai/scal/llm/data/train_action_probs.npy')
action_labels = np.load('/data_hdd/talha/nsai/scal/llm/data/train_action_labels.npy')
top1_preds = np.argmax(action_probs, axis=1)
accuracy = skm.accuracy_score(action_labels, top1_preds)
print(f"Action Prediction Accuracy from saved files: {accuracy*100:.2f}%")


# %%
