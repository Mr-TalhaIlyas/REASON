import numpy as np

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.optim.lr_scheduler import _LRScheduler
import torch.nn.functional as F


class GET_CLIP_OPT_SCHED:
    def __init__(self, clip_lr, clip_params, num_epochs, warmup_epochs, steps_per_epoch, total_steps,
                 warmup_steps):
        self.clip_lr = clip_lr
        self.clip_params = clip_params
        self.num_epochs = num_epochs
        self.warmup_epochs = warmup_epochs
        self.steps_per_epoch = steps_per_epoch
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self._create_clip_optimizer_scheduler()

    def _create_clip_optimizer_scheduler(self):
        self.optimizer_clip = optim.AdamW(
            self.clip_params,
            lr=self.clip_lr, #5e-4,  # 2.5x higher - transformers can handle this
            weight_decay=0.0005,  # Higher weight decay for AdamW
            # betas=(0.9, 0.98)  # Slightly different betas for transformers
        )

        # main_scheduler = lr_scheduler.CosineAnnealingLR(
        #                 self.optimizer_clip,
        #                 T_max=self.total_steps - self.warmup_steps,
        #                 eta_min=1e-5
        #             )
        # warmup_scheduler = lr_scheduler.LinearLR(
        #     self.optimizer_clip,
        #     start_factor=1e-7,
        #     total_iters=self.warmup_steps
        # )
        # self.scheduler_clip = lr_scheduler.SequentialLR(
        #     self.optimizer_clip,
        #     schedulers=[warmup_scheduler, main_scheduler],
        #     milestones=[self.warmup_steps]
        # )
        
        self.scheduler_clip = lr_scheduler.OneCycleLR(
            self.optimizer_clip,
            max_lr=self.clip_lr,  
            total_steps=self.total_steps,
            epochs=self.num_epochs,
            steps_per_epoch=self.steps_per_epoch,
            pct_start=self.warmup_epochs / self.num_epochs, #0.035
            anneal_strategy='cos',
            cycle_momentum=True,
            base_momentum=0.85,
            max_momentum=0.95,
            div_factor=25.0,
            final_div_factor=1e4
        )
        
        # logging
        print(f"Created CLIP optimizer with lr: {self.clip_lr}, weight_decay: 0.01")
        print(f"Optimizer for CLIP: {self.optimizer_clip.__class__.__name__}")
    
class GET_GCN_OPT_SCHED:
    def __init__(self, gcn_lr, gcn_params, num_epochs, warmup_epochs, steps_per_epoch, total_steps,
                 warmup_steps):
        self.gcn_lr = gcn_lr
        self.gcn_params = gcn_params
        self.num_epochs = num_epochs
        self.warmup_epochs = warmup_epochs
        self.steps_per_epoch = steps_per_epoch
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self._create_gcn_optimizer_scheduler()
    
    def _create_gcn_optimizer_scheduler(self):
        # self.optimizer_gcn = optim.SGD(
        #     self.gcn_params,
        #     lr=self.gcn_lr,  # Your stable LR
        #     momentum=0.9,
        #     nesterov=True,
        #     weight_decay=0.0004
        #     )
        self.optimizer_gcn = optim.AdamW(
            self.gcn_params,
            lr=self.gcn_lr, #5e-4,  # 2.5x higher - transformers can handle this
            weight_decay=0.0005,  # Higher weight decay for AdamW
            # betas=(0.9, 0.98)  # Slightly different betas for transformers
        )
        # warmup_schduler = lr_scheduler.LinearLR(
        #     self.optimizer_gcn,
        #     start_factor=1e-7,
        #     total_iters=self.warmup_steps)
        # step_scheduler = lr_scheduler.MultiStepLR(
        #     self.optimizer_gcn,
        #     milestones=[110 * self.steps_per_epoch,
        #                 120 * self.steps_per_epoch],
        #     gamma=0.1
        # )
        # self.scheduler_gcn = lr_scheduler.SequentialLR(
        #     self.optimizer_gcn,
        #     schedulers=[warmup_schduler, step_scheduler],
        #     milestones=[self.warmup_steps]
        # )
        self.scheduler_gcn = lr_scheduler.OneCycleLR(
            self.optimizer_gcn,
            max_lr=self.gcn_lr,  
            total_steps=self.total_steps,
            epochs=self.num_epochs,
            steps_per_epoch=self.steps_per_epoch,
            pct_start=self.warmup_epochs / self.num_epochs, #0.035
            anneal_strategy='cos',
            cycle_momentum=True,
            base_momentum=0.85,
            max_momentum=0.95,
            div_factor=25.0,
            final_div_factor=1e4
        )
        print(f"Created GCN optimizer with lr: {self.gcn_lr}, momentum: 0.9, weight_decay: 0.0004")
        print(f"Optimizer for GCN: {self.optimizer_gcn.__class__.__name__}")

class GET_HEADS_OPT_SCHED:
    def __init__(self, head_lr, head_params, num_epochs, warmup_epochs, steps_per_epoch, total_steps,
                 warmup_steps):
        self.head_lr = head_lr
        self.head_params = head_params
        self.num_epochs = num_epochs
        self.warmup_epochs = warmup_epochs
        self.steps_per_epoch = steps_per_epoch
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self._create_heads_optimizer_scheduler()
    
    def _create_heads_optimizer_scheduler(self):
        self.optimizer_heads = optim.AdamW(
            self.head_params,
            lr=self.head_lr,
            weight_decay=0.0005
        )

        main_scheduler = lr_scheduler.CosineAnnealingLR(
                        self.optimizer_heads,
                        T_max=self.total_steps - self.warmup_steps,
                        eta_min=1e-5
                    )
        warmup_scheduler = lr_scheduler.LinearLR(
            self.optimizer_heads,
            start_factor=1e-7,
            total_iters=self.warmup_steps
        )
        self.scheduler_heads = lr_scheduler.SequentialLR(
            self.optimizer_heads,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[self.warmup_steps]
        )
        print(f"Created Heads optimizer with lr: {self.head_lr}, weight_decay: 0.0004")
        print(f"Optimizer for Heads: {self.optimizer_heads.__class__.__name__}")

def add_weight_decay(model, weight_decay=1e-4, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]
#%%
class StagedLR(_LRScheduler):
    """
    A wrapper scheduler that keeps a parameter group's LR at 0 until a
    specified activation epoch, then applies a separate target LR with step decay.
    This allows other parameter groups to follow their own schedule uninterrupted.
    """
    def __init__(self, optimizer, base_scheduler, logic_group_index, logic_layers_lr,
                  logic_activate_epoch, logic_step_epoch, steps_per_epoch, total_epoch,
                  step_milestones=None, step_decay_rate=0.1, warm_up_epoch_logic=0,
                  logic_lr_type='cosine'):
        self.base_scheduler = base_scheduler
        self.logic_group_index = logic_group_index
        self.logic_layers_lr = logic_layers_lr
        self.logic_activate_epoch = logic_activate_epoch
        self.logic_step_epoch = logic_step_epoch
        self.steps_per_epoch = steps_per_epoch
        self.total_epoch = total_epoch
        self.warm_up_epoch_logic = warm_up_epoch_logic
        self.logic_lr_type = logic_lr_type
        # Step decay parameters
        self.step_milestones = step_milestones if step_milestones is not None else [60, 90, 105]
        self.step_decay_rate = step_decay_rate
        
        super().__init__(optimizer)

    def get_lr(self):
        """Get LRs from the base scheduler (required by _LRScheduler parent)."""
        return self.base_scheduler.get_last_lr()

    def get_cosine_lr(self, epoch, warmup_epoch, total_epochs):
        """
        Cosine annealing schedule with optional warmup.
        Returns a multiplier in [0, 1].
        """
        # Linear warmup
        if epoch < warmup_epoch and warmup_epoch > 0:
            return float(epoch) / float(max(1, warmup_epoch))
        
        # Cosine annealing
        progress = float(epoch - warmup_epoch) / float(max(1, total_epochs - warmup_epoch))
        
        # Ensure progress is within [0, 1]
        progress = max(0.0, min(1.0, progress))
        
        # Cosine schedule from 1.0 down to 0.0
        return 0.5 * (1.0 + np.cos(np.pi * progress))
    
    def get_step_lr(self, epoch, warmup_epoch, step_milestones, decay_rate):
        """
        Step decay schedule with optional warmup.
        Returns a multiplier based on how many milestones have been passed.
        
        Args:
            epoch: Current epoch (relative to activation)
            warmup_epoch: Number of warmup epochs
            step_milestones: List of epochs where LR decays (e.g., [60, 90, 105])
            decay_rate: Multiplicative factor for decay (e.g., 0.1)
        
        Returns:
            float: LR multiplier
        """
        # Linear warmup
        if epoch < warmup_epoch and warmup_epoch > 0:
            return float(epoch + 1) / float(max(1, warmup_epoch))
        
        # Step decay: count how many milestones we've passed
        # Adjust milestones relative to warmup
        adjusted_milestones = [m - warmup_epoch for m in step_milestones if m >= warmup_epoch]
        epoch_after_warmup = epoch - warmup_epoch
        
        # Calculate decay factor
        num_decays = sum(epoch_after_warmup >= milestone for milestone in adjusted_milestones)
        return decay_rate ** num_decays
    
    def step(self, epoch=None):
        """
        Step the scheduler. Updates all parameter groups via base_scheduler,
        then manually overrides the logic layer LR based on activation status.
        """
        # 1. Let the base scheduler update all groups
        self.base_scheduler.step()

        # 2. Calculate current epoch
        current_epoch = self.base_scheduler.last_epoch // self.steps_per_epoch

        # 3. Override logic layer LR based on activation status
        logic_group = self.optimizer.param_groups[self.logic_group_index]
        
        if current_epoch < self.logic_activate_epoch:
            # Before activation: keep at 0
            logic_group['lr'] = 0.0
        else:
            # After activation: apply custom schedule
            epochs_since_activation = current_epoch - self.logic_activate_epoch
            
            if self.logic_lr_type == 'cosine':
                # Option 1: Use cosine schedule (your current implementation)
                lr_multiplier = self.get_cosine_lr(
                    epochs_since_activation, 
                    warmup_epoch=self.warm_up_epoch_logic,
                    total_epochs=self.total_epoch - self.logic_activate_epoch
                )
            elif self.logic_lr_type == 'step':
                # Option 2: Use step decay schedule
                lr_multiplier = self.get_step_lr(
                    epochs_since_activation,
                    warmup_epoch=self.warm_up_epoch_logic,  # No warmup after activation (already trained for 50 epochs)
                    step_milestones=self.step_milestones,
                    decay_rate=self.step_decay_rate
                )
            else:
                raise ValueError(f"Unknown logic_lr_type: {self.logic_lr_type}")
            
            logic_group['lr'] = lr_multiplier * self.logic_layers_lr
            
    def state_dict(self):
        """Returns the state of the scheduler as a dict."""
        return {
            'base_scheduler': self.base_scheduler.state_dict(),
        }

    def load_state_dict(self, state_dict):
        """Loads the scheduler's state."""
        self.base_scheduler.load_state_dict(state_dict['base_scheduler'])

def build_cosine_warmup_scheduler(optimizer, warmup_steps, total_steps):
    """
    Returns a LambdaLR that does linear warmup then cosine annealing.
    The lambda function returns a scaling factor from 0.0 to 1.0.
    """
    def lr_lambda(current_step):
        # Linear warmup
        if current_step < warmup_steps and warmup_steps > 0:
            return float(current_step) / float(max(1, warmup_steps))
        
        # Cosine annealing
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        
        # Ensure progress is within [0, 1]
        progress = max(0.0, min(1.0, progress))
        
        # Cosine schedule from 1.0 down to 0.0
        return 0.5 * (1.0 + np.cos(np.pi * progress))
        
    return lr_scheduler.LambdaLR(optimizer, lr_lambda)




class GET_OPT_SCHED:
    def __init__(self, backbone_lr, head_lr, logic_layers_lr, param_groups,
                 num_epochs, warm_up_epochs, steps_per_epoch, total_steps,
                 warmup_steps, weight_decay, logic_step_epoch, logic_activate_epoch,
                 logic_warmup_epochs, logic_lr_type):
        self.backbone_lr = backbone_lr
        self.logic_layers_lr = logic_layers_lr
        self.head_lr = head_lr
        self.param_groups = param_groups
        self.num_epochs = num_epochs
        self.warm_up_epochs = warm_up_epochs
        self.steps_per_epoch = steps_per_epoch
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.logic_step_epoch = logic_step_epoch
        self.logic_activate_epoch = logic_activate_epoch
        self.logic_warmup_epochs = logic_warmup_epochs
        self.logic_lr_type = logic_lr_type
        self.wd = weight_decay
        self._create_optimizer_scheduler()
    
    def _create_optimizer_scheduler(self):
        # self.optimizer = AdamW(self.model.parameters(), lr=CRL_LR, weight_decay=wd)
        self.optimizer = optim.AdamW(self.param_groups, lr=self.head_lr, weight_decay=self.wd)

        base_scheduler = lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=[self.backbone_lr, self.head_lr, self.logic_layers_lr],  # ✅ CHANGED: Logic layers start at 0
            total_steps=self.total_steps,
            epochs=self.num_epochs,
            steps_per_epoch=self.steps_per_epoch,
            pct_start=self.warm_up_epochs / self.num_epochs, #0.035
            anneal_strategy='cos',
            cycle_momentum=True,
            base_momentum=0.85,
            max_momentum=0.95,
            div_factor=25.0,
            final_div_factor=1e4
        )

        self.scheduler = StagedLR(
            optimizer=self.optimizer,
            base_scheduler=base_scheduler,
            logic_group_index=2,  # The logic group is the 3rd one (index 2)
            logic_layers_lr=self.logic_layers_lr,
            logic_activate_epoch=self.logic_activate_epoch,
            logic_step_epoch=self.logic_step_epoch,
            steps_per_epoch=self.steps_per_epoch,
            total_epoch=self.num_epochs,
            step_milestones=[20, 35, 50],  # Epochs (relative to activation) where LR decays
            step_decay_rate=0.1,  # Multiply LR by 0.1 at each milestone
            warm_up_epoch_logic = self.logic_warmup_epochs,
            logic_lr_type=self.logic_lr_type  # Use cosine schedule for logic layers
        )

