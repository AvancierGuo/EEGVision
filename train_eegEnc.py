import warnings
warnings.filterwarnings('ignore')
import torch.utils.data as Data
from args import args, Test_data, Train_data_all
from dataset import Dataset
from model.EEGVisionModels import TimeEncoder
from process import Trainer
import argparse
import torch; torch.utils.backcompat.broadcast_warning.enabled = True
import torch.optim
import torch.backends.cudnn as cudnn; cudnn.benchmark = True
import numpy as np

parser = argparse.ArgumentParser(description="Template")
parser.add_argument('-mp','--model_params', default='', nargs='*', help='list of key=value pairs of model options')
# Parse arguments
opt = parser.parse_args()

def main():
    torch.set_num_threads(12)
    torch.cuda.manual_seed(3407)
    train_dataset = Dataset(device=args.device, mode='pretrain', data=Train_data_all, wave_len=args.wave_length)
    test_dataset = Dataset(device=args.device, mode='test', data=Test_data, wave_len=args.wave_length)
    args.data_shape = train_dataset.shape()
    print("数据形状:",train_dataset.shape())
    print("训练集数量:",len(train_dataset))
    print("测试集数量:",len(test_dataset))
    train_loader = Data.DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True,drop_last=False)
    train_linear_dataset = Dataset(device=args.device, mode='supervise_train', data=Train_data_all, wave_len=args.wave_length)
    train_linear_loader = Data.DataLoader(train_linear_dataset, batch_size=args.train_batch_size, shuffle=True,drop_last=False)
    test_loader = Data.DataLoader(test_dataset, batch_size=args.test_batch_size,drop_last=False)

    time_model = TimeEncoder(args)
    trainer = Trainer(args, time_model, train_loader, train_linear_loader, test_loader, verbose=True)

    trainer.pretrain()
    # trainer.finetune()

    ## Start from this step, to finetune on single subject, please modify the 'datautils.py'.
    # trainer.finetune_timefreq()
    # trainer.finetune_CLIP()

if __name__ == '__main__':
    main()
