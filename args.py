import argparse
import os
import json
import torch

parser = argparse.ArgumentParser()
# dataset and dataloader args
parser.add_argument('--save_path', type=str, default='exp/epilepsy/test')
parser.add_argument('--dataset', type=str, default='eeg')
parser.add_argument('--data_path', type=str,
                    default='data/EEG/')
#parser.add_argument('--device', type=str, default='cpu')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--train_batch_size', type=int, default=128)
parser.add_argument('--test_batch_size', type=int, default=128)

# model args
parser.add_argument('--d_model', type=int, default=1024)
parser.add_argument('--dropout', type=float, default=0)
parser.add_argument('--attn_heads', type=int, default=16)
parser.add_argument('--eval_per_steps', type=int, default=16)
parser.add_argument('--enable_res_parameter', type=int, default=1)
parser.add_argument('--layers', type=int, default=8)
parser.add_argument('--alpha', type=float, default=4.0)
parser.add_argument('--beta', type=float, default=2.0)

parser.add_argument('--momentum', type=float, default=0.99)
parser.add_argument('--vocab_size', type=int,  default=660)
parser.add_argument('--wave_length', type=int, default=4)
parser.add_argument('--mask_ratio', type=float, default=0.75)
parser.add_argument('--reg_layers', type=int, default=4)

# train args
parser.add_argument('--lr', type=float, default=0.0005)
parser.add_argument('--lr_decay_rate', type=float, default=1.)
parser.add_argument('--lr_decay_steps', type=int, default=100)
parser.add_argument('--weight_decay', type=float, default=0.01)
parser.add_argument('--num_epoch_pretrain', type=int, default=300)
parser.add_argument('--num_epoch', type=int, default=80)
parser.add_argument('--load_pretrained_model', type=int, default=1)

args = parser.parse_args()

if args.data_path is None:
    if args.dataset == 'eeg':
        # Train_data_all, Train_data, Test_data = load_EEG()
        f = open("data/EEG_divided/Subj_01_eeg_label_clip/all_train_data.pkl", "rb")
        Train_data_all = torch.load(f,map_location='cpu')
        f.close()
        f = open("data/EEG_divided/Subj_01_eeg_label_clip/train_data.pkl", "rb")
        Train_data = torch.load(f,map_location='cpu')
        f.close()
        f = open("data/EEG_divided/Subj_01_eeg_label_clip/test_data.pkl", "rb")
        Test_data = torch.load(f,map_location='cpu')
        f.close()

        args.num_class = len(set(Train_data[1]))
else:
    if args.dataset == 'eeg':
        path = args.data_path
        # Train_data_all, Train_data, Test_data = load_EEG()
        f = open("data/EEG_divided/Subj_01_eeg_label_clip/all_train_data.pkl", "rb")
        Train_data_all = torch.load(f,map_location='cpu')
        f.close()
        f = open("data/EEG_divided/Subj_01_eeg_label_clip/train_data.pkl", "rb")
        Train_data = torch.load(f,map_location='cpu')
        f.close()
        f = open("data/EEG_divided/Subj_01_eeg_label_clip/test_data.pkl", "rb")
        Test_data = torch.load(f,map_location='cpu')
        f.close()
        
        args.num_class = len(set(Train_data[1]))

args.eval_per_steps = max(1, int(len(Train_data[0]) / args.train_batch_size))
args.lr_decay_steps = args.eval_per_steps
if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)
config_file = open(args.save_path + '/args.json', 'w')
tmp = args.__dict__
json.dump(tmp, config_file, indent=1)
print(args)
config_file.close()
