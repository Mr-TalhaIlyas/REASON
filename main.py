#!/usr/bin/env python
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

import torch
import torch.nn.functional as F
from optim_utils import GET_OPT_SCHED

import wandb
from torchmetrics import Accuracy
from torchmetrics.classification import MultilabelExactMatch, MultilabelAccuracy

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
from model.model import SkeletonACL_CLIP_Logic



def init_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


class Trainer:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.args = yaml.safe_load(f)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        init_seed(self.args.get('seed', 1))

        # Initialize WandB
        self.init_wandb()

        # dataloaders
        self.data_loader = {}
        self.data_loader['train'] = torch.utils.data.DataLoader(
            dataset=Feeder(**self.args['train_feeder_args']),
            batch_size=self.args['batch_size'],
            shuffle=True,
            num_workers=self.args['num_worker'],
            drop_last=True,
            collate_fn=custom_collate_fn,
            worker_init_fn=lambda _: init_seed(self.args.get('seed', 1)),
            pin_memory=True,
            persistent_workers=True if self.args['num_worker'] > 0 else False
        )
        self.data_loader['test'] = torch.utils.data.DataLoader(
            dataset=Feeder(**self.args['test_feeder_args']),
            batch_size=self.args['test_batch_size'],
            shuffle=False,
            num_workers=self.args['num_worker'],
            drop_last=False,
            collate_fn=custom_collate_fn,
            worker_init_fn=lambda _: init_seed(self.args.get('seed', 1)),
            pin_memory=True,
            persistent_workers=True if self.args['num_worker'] > 0 else False
        )

        # model
        self.model = SkeletonACL_CLIP_Logic(self.args, self.device).to(self.device)
        '''
        NEW
        '''
        # self.backbone_lr = 1e-5  # Small LR for pre-trained models
        self.clip_lr = 1e-5 # 5e-5 5e-4
        self.gcn_lr = 1e-5 #0.02 0.05
        self.head_lr = 1e-4 #1e-3 # Larger LR for new layers
        self.logic_layers_lr = 1e-4  # Start with 0 LR for logic layers
        self.use_concept_refiner = self.args['use_concept_refiner']
        self.logic_activate_epoch = int(self.args.get('logic_activate_epoch', 0)) 
        self.logic_step_epoch = int(self.args.get('logic_step_epoch', 15)) 

        '''
        GET OPTIMIZERS AND SCHEDULERS
        '''
        # scheduler: cosine with warmup
        self.num_epochs = int(self.args.get('num_epoch', 110))
        self.warm_up_epochs = int(self.args.get('warm_up_epoch', 5))
        self.steps_per_epoch = len(self.data_loader['train'])
        self.total_steps = self.num_epochs * self.steps_per_epoch
        self.warmup_steps = self.warm_up_epochs * self.steps_per_epoch
        self.logic_warmup_epochs = int(self.args.get('warm_up_epoch_logic', 0)) # NEW
        self.logic_lr_type = self.args.get('logic_lr_type', 'cosine')  # 'step' or 'cosine'
        wd = float(self.args.get('weight_decay', 0.0005))
        
        backbone_params = list(self.model.skeleton_model.parameters()) + \
                        list(self.model.text_encoder.parameters())

        head_params = list(self.model.skel_projection.parameters()) + \
                    list(self.model.text_projection.parameters()) + \
                    list(self.model.fc.parameters())    
        if self.use_concept_refiner:
            head_params += list(self.model.concept_refiner.parameters())
             
        logic_head_params = list(self.model.logic_layers.parameters())
        # Include the learnable logit_scale with the heads
        head_params.append(self.model.mp_clip_loss.logit_scale)
        
        param_groups = [
            {'params': backbone_params, 'lr': self.gcn_lr, 'weight_decay': wd, 'name': 'backbone'},
            {'params': head_params, 'lr': self.head_lr, 'weight_decay': wd, 'name': 'heads'},
            {'params': logic_head_params, 'lr': self.logic_layers_lr, 'weight_decay': 0, 'name': 'logic_head'}  # CHANGED: Start with LR=0
        ]
        
        opt_sched = GET_OPT_SCHED(
            backbone_lr=self.gcn_lr,
            head_lr=self.head_lr,
            logic_layers_lr=self.logic_layers_lr,
            param_groups=param_groups,
            num_epochs=self.num_epochs,
            warm_up_epochs=self.warm_up_epochs,
            steps_per_epoch=self.steps_per_epoch,
            total_steps=self.total_steps,
            warmup_steps=self.warmup_steps,
            weight_decay = wd,
            logic_step_epoch=self.logic_step_epoch,
            logic_activate_epoch=self.logic_activate_epoch,
            logic_warmup_epochs=self.logic_warmup_epochs,
            logic_lr_type=self.logic_lr_type
        )
        self.optimizer, self.scheduler = opt_sched.optimizer, opt_sched.scheduler
        
        # mixed precision scaler
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.args.get('use_amp', True))

        # metrics
        self.init_metrics()

        # bookkeeping
        self.work_dir = Path(self.args['work_dir'])
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.best_acc = 0.0
        self.best_epoch = 0
        self.current_exact = 0.0

        print("\n" + "="*80)
        print("✅ TRAINER INITIALIZED (AdamW + OneCycleLR + Staged Logic Training)")
        print("="*80)
        print(f"Work directory: {self.work_dir}")
        print(f"Device: {self.device}")
        print(f"Total epochs: {self.num_epochs}")
        print(f"Batch size: {self.args['batch_size']}")
        print(f"GCN LR: {self.gcn_lr:.2e}")
        print(f"CLIP LR: {self.clip_lr:.2e}")
        print(f"Head LR: {self.head_lr:.2e}")
        print(f"Logic LR (initial): {self.logic_layers_lr:.2e}")
        print(f"AMP enabled: {self.args.get('use_amp', True)}")
        print("="*80 + "\n")
        print(f"Head LR: {self.head_lr:.2e}")
        print(f"Logic LR (initial): 0.0 (activates at epoch {self.logic_activate_epoch})") # ✅ Updated print
        print(f"AMP enabled: {self.args.get('use_amp', True)}")
    


    def init_wandb(self):
        """Initialize Weights & Biases logging."""
        if self.args.get('use_wandb', False):
            wandb.init(
                project=self.args.get('wandb_project', 'SkeletonACL'),
                entity=self.args.get('wandb_entity', None),
                name=self.args.get('wandb_name', 'skeleton_acl_single_opt'),
                config=self.args,
                resume='allow'
            )
            print("✅ WandB initialized")
        else:
            print("⚠️  WandB logging disabled")
    
    def init_metrics(self):
        """Initialize TorchMetrics for tracking accuracy."""
        model_args = self.args.get('model_args', {})
        self.num_actions = 120  # NTU-120
        self.num_spatial = int(model_args.get('spatial_concepts', 52))
        self.num_temporal = int(model_args.get('temporal_concepts', 26))

        # Action metrics
        self.train_action_acc = Accuracy(task='multiclass', num_classes=self.num_actions).to(self.device)
        self.train_action_top5 = Accuracy(task='multiclass', num_classes=self.num_actions, top_k=5).to(self.device)
        self.val_action_acc = Accuracy(task='multiclass', num_classes=self.num_actions).to(self.device)
        self.val_action_top5 = Accuracy(task='multiclass', num_classes=self.num_actions, top_k=5).to(self.device)

        # Spatial concept metrics
        self.train_spatial_exact = MultilabelExactMatch(num_labels=self.num_spatial).to(self.device)
        self.train_spatial_acc = MultilabelAccuracy(num_labels=self.num_spatial, average='micro').to(self.device)
        self.val_spatial_exact = MultilabelExactMatch(num_labels=self.num_spatial).to(self.device)
        self.val_spatial_acc = MultilabelAccuracy(num_labels=self.num_spatial, average='micro').to(self.device)

        # Temporal concept metrics
        self.train_temporal_exact = MultilabelExactMatch(num_labels=self.num_temporal).to(self.device)
        self.train_temporal_acc = MultilabelAccuracy(num_labels=self.num_temporal, average='micro').to(self.device)
        self.val_temporal_exact = MultilabelExactMatch(num_labels=self.num_temporal).to(self.device)
        self.val_temporal_acc = MultilabelAccuracy(num_labels=self.num_temporal, average='micro').to(self.device)
        # Overall concept exact match
        self.val_exact = MultilabelExactMatch(num_labels=self.num_spatial + self.num_temporal).to(self.device)  

        print("✅ TorchMetrics initialized:")
        print(f"   • Action classes: {self.num_actions}")
        print(f"   • Spatial concepts: {self.num_spatial}")
        print(f"   • Temporal concepts: {self.num_temporal}\n")

    def reset_train_metrics(self):
        """Reset training metrics at the start of each epoch."""
        self.train_action_acc.reset()
        self.train_action_top5.reset()
        self.train_spatial_exact.reset()
        self.train_spatial_acc.reset()
        self.train_temporal_exact.reset()
        self.train_temporal_acc.reset()

    def reset_val_metrics(self):
        """Reset validation metrics at the start of evaluation."""
        self.val_action_acc.reset()
        self.val_action_top5.reset()
        self.val_spatial_exact.reset()
        self.val_spatial_acc.reset()
        self.val_temporal_exact.reset()
        self.val_temporal_acc.reset()
        self.val_exact.reset()

    def train_epoch(self, epoch):
        """Train for one epoch with mixed precision."""
        self.model.train()
        self.reset_train_metrics()
            
        epoch_loss = 0.0
        epoch_action_loss = 0.0
        epoch_concept_loss = 0.0
        epoch_contrastive_loss = 0.0
        epoch_l2_loss = 0.0
        epoch_div_loss = 0.0
        
        pbar = tqdm(
            enumerate(self.data_loader['train']),
            total=len(self.data_loader['train']),
            desc=f"Epoch {epoch}/{self.args['num_epoch']} [TRAIN]",
            ncols=140
        )
        
        for step, (batch_data, batch_label, batch_concept_vecs, batch_prompts) in pbar:
            batch_data = batch_data.to(self.device)
            batch_label = batch_label.to(self.device)
            s_trgt = batch_concept_vecs['full_body'].to(self.device)
            t_trgt = batch_concept_vecs['temporal'].to(self.device)

            # if step == 0:
            #     self.model.update_temperature(epoch, self.num_epochs)
                
            self.optimizer.zero_grad()
            
            with torch.cuda.amp.autocast(enabled=self.args.get('use_amp', True)):
                # Model returns: (action_logits, spatial_probs, temporal_probs, loss_dict)
                y_pred, s_probs, t_probs, loss_dict = self.model(
                    batch_data, batch_label, batch_concept_vecs, batch_prompts,
                    epoch, self.num_epochs, self.warm_up_epochs
                )
            
            # Backward pass with gradient scaling
            self.scaler.scale(loss_dict['total_loss']).backward()
            
            # Gradient clipping (single optimizer)
            self.scaler.unscale_(self.optimizer)
            
            
            # FIX 2: Add per-group gradient clipping with stricter control for logic layers
            # Clip backbone and heads with max_norm=1.0
            torch.nn.utils.clip_grad_norm_(self.optimizer.param_groups[0]['params'], max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(self.optimizer.param_groups[1]['params'], max_norm=1.0) # heads
            torch.nn.utils.clip_grad_norm_(self.optimizer.param_groups[2]['params'], max_norm=0.5) # logic layers
            
            # Calculate total gradient norm for logging
            trainable_params = []
            for group in self.optimizer.param_groups:
                trainable_params.extend(group['params'])
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=float('inf'))  # Just for measurement
            # get gcn grad norm for logging
            gcn_grad_norm = torch.nn.utils.clip_grad_norm_(self.optimizer.param_groups[0]['params'], max_norm=float('inf'))
            clip_grad_norm = torch.nn.utils.clip_grad_norm_(self.optimizer.param_groups[0]['params'], max_norm=float('inf'))
            heads_grad_norm = torch.nn.utils.clip_grad_norm_(self.optimizer.param_groups[1]['params'], max_norm=float('inf'))
            logic_grad_norm = torch.nn.utils.clip_grad_norm_(self.optimizer.param_groups[2]['params'], max_norm=float('inf'))
            # FIX 2b: Add NaN/Inf detection and recovery
            if not torch.isfinite(grad_norm):
                print(f"\n⚠️  WARNING: NaN/Inf gradient detected at batch {step}")
                print(f"   Zeroing gradients and skipping step...\n")
                self.optimizer.zero_grad()
                self.scaler.update()
                continue
            # Check for gradient explosion
            if grad_norm > 100:
                print(f"\n⚠️  WARNING: Gradient explosion at batch {step} (norm={grad_norm:.2f})")
                print(f"   Skipping step...\n")
                self.optimizer.zero_grad()
                self.scaler.update()
                continue
            
            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()
                
            # Step scheduler by global step
            global_step = epoch * len(self.data_loader['train']) + step
            self.scheduler.step()
            
            # Clip logic layer weights only when they are being trained
            with torch.no_grad():
                self.model.logic_layers.clip_weights()

            # Accumulate losses
            epoch_loss += loss_dict['total_loss'].item()
            epoch_action_loss += loss_dict['action_loss'].item()
            epoch_concept_loss += loss_dict['concept_loss'].item()
            epoch_contrastive_loss += loss_dict['contrastive_loss'].item()
            epoch_l2_loss += loss_dict['l2_penalty_loss'].item()
            epoch_div_loss += loss_dict['div_loss'].item()

            # ✅ Update metrics
            self.train_action_acc.update(y_pred, batch_label)
            self.train_action_top5.update(y_pred, batch_label)

            # Multi-label predictions (threshold at 0.5)
            s_pred = (s_probs > 0.5).long()
            t_pred = (t_probs > 0.5).long()

            self.train_spatial_exact.update(s_pred, s_trgt.long())
            self.train_spatial_acc.update(s_pred, s_trgt.long())
            self.train_temporal_exact.update(t_pred, t_trgt.long())
            self.train_temporal_acc.update(t_pred, t_trgt.long())

            # Progress display
            current_action_acc = self.train_action_acc.compute().item() * 100
            current_spatial_exact = self.train_spatial_exact.compute().item() * 100
            current_temporal_exact = self.train_temporal_exact.compute().item() * 100
            
            if self.use_concept_refiner:
                refiner_alpha = self.model.concept_refiner.alpha.item()
            else:
                refiner_alpha = 0.0
            pbar.set_postfix({
                'loss': f"{loss_dict['total_loss'].item():.4f}",
                'a_acc': f"{current_action_acc:.2f}%",
                's_exact': f"{current_spatial_exact:.2f}%",
                't_exact': f"{current_temporal_exact:.2f}%",
                'g_norm': f"{grad_norm:.2f}"
            })
            
            # Log to WandB - Organized into 5 main groups
            if self.args.get('use_wandb', False) and step % self.args.get('log_interval', 100) == 0:
                wandb.log({
                    # Group 1: Training Losses
                    'train/loss/total': loss_dict['total_loss'].item(),
                    'train/loss/action': loss_dict['action_loss'].item(),
                    'train/loss/concept': loss_dict['concept_loss'].item(),
                    'train/loss/contrastive': loss_dict['contrastive_loss'].item(),
                    'train/loss/l2_penalty': loss_dict['l2_penalty_loss'].item(),
                    'train/loss/divergence': loss_dict['div_loss'].item(),
                    
                    # Group 2: Training Accuracy
                    'train/accuracy/action_top1': current_action_acc,
                    'train/accuracy/spatial_exact': current_spatial_exact,
                    'train/accuracy/temporal_exact': current_temporal_exact,
                    
                    # Group 3: Gradient Norms
                    'grad_norm/total': grad_norm.item(),
                    'grad_norm/gcn': gcn_grad_norm.item(),
                    'grad_norm/clip': clip_grad_norm.item(),
                    'grad_norm/heads': heads_grad_norm.item(),
                    'grad_norm/logic': logic_grad_norm.item(),
                    
                    # Group 4: Learning Rates
                    'lr/gcn': self.optimizer.param_groups[0]['lr'],
                    # 'lr/clip': self.optimizer_clip.param_groups[0]['lr'],
                    'lr/heads': self.optimizer.param_groups[1]['lr'],
                    'lr/logic': self.optimizer.param_groups[2]['lr'],
                    'lr/RefinerAlpha': refiner_alpha
                }, step=global_step)
            # break  # TEMPORARY: REMOVE THIS LINE TO RUN FULL EPOCH

        # ✅ Compute final epoch metrics
        num_batches = len(self.data_loader['train'])
        epoch_loss /= num_batches
        epoch_action_loss /= num_batches
        epoch_concept_loss /= num_batches
        epoch_contrastive_loss /= num_batches
        epoch_l2_loss /= num_batches
        epoch_div_loss /= num_batches
        
        action_acc = self.train_action_acc.compute().item() * 100
        action_top5 = self.train_action_top5.compute().item() * 100
        spatial_exact = self.train_spatial_exact.compute().item() * 100
        spatial_acc = self.train_spatial_acc.compute().item() * 100
        temporal_exact = self.train_temporal_exact.compute().item() * 100
        temporal_acc = self.train_temporal_acc.compute().item() * 100

        # Print epoch summary
        print(f"\n{'='*80}")
        print(f"EPOCH {epoch} TRAINING SUMMARY")
        print(f"{'='*80}")
        print(f"Losses:")
        print(f"  Total:         {epoch_loss:.4f}")
        print(f"  Action:        {epoch_action_loss:.4f}")
        print(f"  Concept:       {epoch_concept_loss:.4f}")
        print(f"  Contrastive:   {epoch_contrastive_loss:.4f}")
        print(f"  L2 Penalty:    {epoch_l2_loss:.6f}")
        print(f"  Divergence:    {epoch_div_loss:.6f}")
        print(f"\nAction Metrics:")
        print(f"  Top-1 Acc:     {action_acc:.2f}%")
        print(f"  Top-5 Acc:     {action_top5:.2f}%")
        print(f"\nSpatial Concept Metrics:")
        print(f"  Exact Match:   {spatial_exact:.2f}%")
        print(f"  Per-Class:     {spatial_acc:.2f}%")
        print(f"\nTemporal Concept Metrics:")
        print(f"  Exact Match:   {temporal_exact:.2f}%")
        print(f"  Per-Class:     {temporal_acc:.2f}%")
        print(f"\nLearning Rate (GCN): {self.optimizer.param_groups[0]['lr']:.6f}")
        print(f"Learning Rate (CLIP): {self.optimizer.param_groups[0]['lr']:.6f}")
        print(f"Learning Rate (Head):     {self.optimizer.param_groups[1]['lr']:.6f}")
        print(f"Learning Rate (Logic):     {self.optimizer.param_groups[2]['lr']:.6f}")
        print(f"{'='*80}\n")

        return epoch_loss, action_acc

    @torch.no_grad()
    def evaluate(self, epoch, split='test'):
        """Evaluate on test/validation set."""
        self.model.eval()
        self.reset_val_metrics()
        
        pbar = tqdm(
            self.data_loader[split],
            desc=f"Epoch {epoch}/{self.args['num_epoch']} [EVAL]",
            ncols=120
        )
        
        for batch_data, batch_label, batch_concept_vecs, batch_prompts in pbar:
            batch_data = batch_data.to(self.device)
            batch_label = batch_label.to(self.device)
            s_trgt = batch_concept_vecs['full_body'].to(self.device)
            t_trgt = batch_concept_vecs['temporal'].to(self.device)

            with torch.cuda.amp.autocast(enabled=self.args.get('use_amp', True)):
                with torch.inference_mode():
                    # Model returns: (action_logits, spatial_probs, temporal_probs, loss_dict)
                    y_pred, s_probs, t_probs, _ = self.model(
                        batch_data, batch_label, batch_concept_vecs, batch_prompts)

            # Update metrics
            self.val_action_acc.update(y_pred, batch_label)
            self.val_action_top5.update(y_pred, batch_label)
            
            s_pred = (s_probs > 0.5).long()
            t_pred = (t_probs > 0.5).long()

            self.val_spatial_exact.update(s_pred, s_trgt.long())
            self.val_spatial_acc.update(s_pred, s_trgt.long())
            self.val_temporal_exact.update(t_pred, t_trgt.long())
            self.val_temporal_acc.update(t_pred, t_trgt.long())
            # NEW : overall exact match
            full_pred = torch.cat([s_pred, t_pred], dim=1)
            full_trgt = torch.cat([s_trgt.long(), t_trgt.long()], dim=1)
            self.val_exact.update(full_pred, full_trgt)
            
            # Update progress bar
            current_action_acc = self.val_action_acc.compute().item() * 100
            current_spatial_exact = self.val_spatial_exact.compute().item() * 100
            current_temporal_exact = self.val_temporal_exact.compute().item() * 100
            current_exact = self.val_exact.compute().item() * 100
            
            pbar.set_postfix({
                'a_acc': f"{current_action_acc:.2f}%",
                's_exact': f"{current_spatial_exact:.2f}%",
                't_exact': f"{current_temporal_exact:.2f}%",
                'full_exact': f"{current_exact:.2f}%"
            })
            # break  # TEMPORARY: REMOVE THIS LINE TO RUN FULL EPOCH
        # Compute final metrics
        action_top1 = self.val_action_acc.compute().item() * 100
        action_top5 = self.val_action_top5.compute().item() * 100
        spatial_exact = self.val_spatial_exact.compute().item() * 100
        spatial_acc = self.val_spatial_acc.compute().item() * 100
        temporal_exact = self.val_temporal_exact.compute().item() * 100
        temporal_acc = self.val_temporal_acc.compute().item() * 100
        overall_exact = self.val_exact.compute().item() * 100  # NEW

        # Print evaluation summary
        print(f"\n{'='*80}")
        print(f"EPOCH {epoch} EVALUATION SUMMARY ({split.upper()})")
        print(f"{'='*80}")
        print(f"Action Metrics:")
        print(f"  Top-1 Acc:     {action_top1:.2f}%")
        print(f"  Top-5 Acc:     {action_top5:.2f}%")
        print(f"\nSpatial Concept Metrics:")
        print(f"  Exact Match:   {spatial_exact:.2f}%")
        print(f"  Per-Class:     {spatial_acc:.2f}%")
        print(f"\nTemporal Concept Metrics:")
        print(f"  Exact Match:   {temporal_exact:.2f}%")
        print(f"  Per-Class:     {temporal_acc:.2f}%")
        print(f"\nOverall Concept Metrics:")
        print(f"  Exact Match:   {overall_exact:.2f}%")
        print(f"{'='*80}\n")

        # Log to WandB - Test Accuracy Group
        if self.args.get('use_wandb', False):
            wandb.log({
                # Group 5: Test Accuracy
                'test/accuracy/action_top1': action_top1,
                'test/accuracy/action_top5': action_top5,
                'test/accuracy/spatial_exact': spatial_exact,
                'test/accuracy/spatial_per_class': spatial_acc,
                'test/accuracy/temporal_exact': temporal_exact,
                'test/accuracy/temporal_per_class': temporal_acc,
                'test/accuracy/overall_exact': overall_exact,
                'epoch': epoch
            }) # , step=epoch * len(self.data_loader['train'])

        return action_top1, overall_exact

    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint."""
        ckpt = self.work_dir / f"checkpoint_epoch_{epoch}.pth"
        torch.save({
            'epoch': epoch,
            'model_state': self.model.state_dict(),
            # 'optimizer_state': self.optimizer.state_dict(),
            # 'scheduler_state': self.scheduler.state_dict(),
        }, ckpt)
        
        if is_best:
            best_path = self.work_dir / "best_model.pth"
            torch.save({
                'epoch': epoch,
                'model_state': self.model.state_dict(),
                # 'optimizer_state': self.optimizer.state_dict(),
                # 'scheduler_state': self.scheduler.state_dict(),
                'best_acc': self.best_acc,
                'best_epoch': self.best_epoch
            }, best_path)
            print(f"✅ New best model saved! Accuracy: {self.best_acc:.2f}%\n")

    def train(self):
        """Complete training loop."""
        print("\n" + "="*80)
        print("🚀 STARTING TRAINING")
        print("="*80 + "\n")
        
        start_epoch = int(self.args.get('start_epoch', 0))
        num_epochs = int(self.args.get('num_epoch', 110))

        for epoch in range(start_epoch, num_epochs):
            t0 = time.time()
            
            # Training
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Evaluation
            if (epoch + 1) % int(self.args.get('eval_interval', 1)) == 0:
                action_acc, overall_exact = self.evaluate(epoch, split='test')
                
                is_best = action_acc > self.best_acc
                if is_best:
                    self.best_acc = action_acc
                    self.best_epoch = epoch
                    self.current_exact = overall_exact
                
                if (epoch + 1) % int(self.args.get('save_interval', 2)) == 0 or is_best:
                    self.save_checkpoint(epoch, is_best=is_best)
            
            t1 = time.time()
            print(f"⏱️  Epoch {epoch} done in {t1 - t0:.1f}s. Best Action Acc {self.best_acc:.2f}%  concept acc. ({self.current_exact:.2f}%) (epoch {self.best_epoch})\n")
            
            if self.args.get('use_wandb', False):
                wandb.log({
                    'test/accuracy/best_action_top1': self.best_acc,
                    'test/accuracy/best_overall_exact': self.current_exact,
                    'epoch': epoch
                }) # , step=epoch * len(self.data_loader['train'])
        
        print("\n" + "="*80)
        print("🎉 TRAINING COMPLETE!")
        print("="*80)
        print(f"Best Test Accuracy: {self.best_acc:.2f}% (Epoch {self.best_epoch}); Concept Accuracy: {self.current_exact:.2f}%")
        print(f"Final model saved to: {self.work_dir}")
        print("="*80 + "\n")
        
        if self.args.get('use_wandb', False):
            wandb.run.summary['best_test_acc'] = self.best_acc
            wandb.run.summary['best_epoch'] = self.best_epoch
            wandb.finish()

def main():
    """Main entry point."""
    cfg = os.path.join(os.path.dirname(__file__), 'config', 'xset', 'config_skeleton_crl_jm.yaml')
    print(30*'=')
    print(f"Using config file: {cfg}")
    print(30*'~')
    trainer = Trainer(cfg)
    trainer.train()

if __name__ == '__main__':
    main()