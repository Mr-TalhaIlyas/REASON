# import argparse

# # def get_arguments():
# def add_lora_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--seed', default=1, type=int)
#     # Dataset arguments
#     parser.add_argument('--root_path', type=str, default='')
#     parser.add_argument('--dataset', type=str, default='imagenet')
#     parser.add_argument('--shots', default=16, type=int)
#     # Model arguments
#     parser.add_argument('--backbone', default='ViT-B/32', type=str)
#     # Training arguments
#     parser.add_argument('--lr', default=2e-4, type=float)
#     parser.add_argument('--n_iters', default=500, type=int)
#     parser.add_argument('--batch_size', default=32, type=int)
#     # LoRA arguments
#     parser.add_argument('--position', type=str, default='all', choices=['bottom', 'mid', 'up', 'half-up', 'half-bottom', 'all', 'top3'], help='where to put the LoRA modules')
#     parser.add_argument('--encoder', type=str, choices=['text', 'vision', 'both'], default='both')
#     parser.add_argument('--params', metavar='N', type=str, nargs='+', default=['q', 'k', 'v'], help='list of attention matrices where putting a LoRA') 
#     parser.add_argument('--r', default=4, type=int, help='the rank of the low-rank matrices')
#     parser.add_argument('--alpha', default=2, type=int, help='scaling (see LoRA paper)')
#     parser.add_argument('--dropout_rate', default=0.25, type=float, help='dropout rate applied before the LoRA module')

#     parser.add_argument('--save_path', default='/data_hdd/talha/nsai/GAP/work_dir/lora/', help='path to save the lora modules after training, not saved if None')
#     parser.add_argument('--filename', default='lora_weights', help='file name to save the lora weights (.pt extension will be added)')

#     parser.add_argument('--eval_only', default=False, action='store_true', help='only evaluate the LoRA modules (save_path should not be None)')
#     # args = parser.parse_args()

#     return parser

# # lora_args = get_arguments()

from types import SimpleNamespace

def get_lora_args():
    args_dict = {
        'seed': 1,
        'root_path': '',
        'dataset': 'imagenet',
        'shots': 16,
        'backbone': 'ViT-B/32',
        'lr': 0.01, # 2e-4
        'n_iters': 500,
        'batch_size': 32,
        'position': 'all',
        'encoder': 'both',
        'params': ['q', 'k'], # 'v', 'o'
        'r': 16, # 4
        'alpha': 32, # 2
        'dropout_rate': 0.1,
        'save_path': '/data_hdd/talha/nsai/GAP/work_dir/lora/',
        'filename': 'lora_weights',
        'eval_only': False,
        'gcn_lr': 0.05,
        'clip_lr': 5e-4,
    }

    return SimpleNamespace(**args_dict)
