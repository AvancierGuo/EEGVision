import torch; torch.utils.backcompat.broadcast_warning.enabled = True
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim
import numpy as np
from model.EEGVisionModels import FreqEncoder

eeg_dataset ='data/EEG/eeg_5_95_std.pth'
splits_path ='data/EEG/block_splits_by_image_all.pth'
split_num = 0 #leave this always to zero.
subject = 0 #choose a subject from 1 to 6, default is 0 (all subjects)
time_low = 20
time_high = 460
batch_size = 128
optim = "Adam"
learning_rate = 0.001
learning_rate_decay_by = 0.5
learning_rate_decay_every = 10
epochs = 1000
no_cuda = False

class EEGDataset:
    def __init__(self, eeg_signals_path):
        # Load EEG signals
        loaded = torch.load(eeg_signals_path)
        if subject!=0:
            self.data = [loaded['dataset'][i] for i in range(len(loaded['dataset']) ) if loaded['dataset'][i]['subject']==subject]
        else:
            self.data=loaded['dataset']        
        self.labels = loaded["labels"]
        self.images = loaded["images"]
        
        # Compute size
        self.size = len(self.data)

    # Get size
    def __len__(self):
        return self.size

    # Get item
    def __getitem__(self, i):
        # Process EEG
        eeg = self.data[i]["eeg"].float().t()
        eeg = eeg[time_low:time_high,:]

        # # Get label
        label = self.data[i]["label"]
        return eeg, label
    
class Splitter:
    def __init__(self, dataset, split_path, split_num=0, split_name="train"):
        # Set EEG dataset
        self.dataset = dataset
        # Load split
        loaded = torch.load(split_path)
        self.split_idx = loaded["splits"][split_num][split_name]
        # Filter data
        self.split_idx = [i for i in self.split_idx if 450 <= self.dataset.data[i]["eeg"].size(1) <= 600]
        # Compute size
        self.size = len(self.split_idx)

    # Get size
    def __len__(self):
        return self.size

    # Get item
    def __getitem__(self, i):
        # Get sample from dataset
        eeg, label = self.dataset[self.split_idx[i]]
        # Return
        return eeg, label

# Load dataset
dataset = EEGDataset(eeg_dataset)
# Create loaders
loaders = {split: DataLoader(Splitter(dataset, split_path = splits_path, split_num = split_num, split_name = split), batch_size = batch_size, drop_last = True, shuffle = True) for split in ["train", "val", "test"]}
train_dataset=Splitter(dataset, split_path = splits_path, split_num = split_num, split_name = "train")
print("train_num:",len(train_dataset))

# Load model
model = FreqEncoder()
optimizer = getattr(torch.optim, optim)(model.parameters(), lr = learning_rate)
    
# Setup CUDA
if not no_cuda:
    model.cuda()

#initialize training,validation, test losses and accuracy list
losses_per_epoch={"train":[], "val":[],"test":[]}
accuracies_per_epoch={"train":[],"val":[],"test":[]}

best_accuracy = 0
best_accuracy_val = 0
best_epoch = 0

predicted_labels = [] 
correct_labels = []

for epoch in range(1, epochs+1):
    # Initialize loss/accuracy variables
    losses = {"train": 0, "val": 0, "test": 0}
    accuracies = {"train": 0, "val": 0, "test": 0}
    counts = {"train": 0, "val": 0, "test": 0}
    # Process each split
    for split in ("train", "val", "test"):
        # Set network mode
        if split == "train":
            model.train()
            torch.set_grad_enabled(True)
        else:
            model.eval()
            torch.set_grad_enabled(False)
        # Process all split batches
        for i, (input, target) in enumerate(loaders[split]):
            # Check CUDA
            if not no_cuda:
                input = input.to("cuda") 
                target = target.to("cuda")

            # Forward
            output,xa = model(input)
            # Compute loss
            loss = F.cross_entropy(output, target)
            losses[split] += loss.item()
            # Compute accuracy
            _,pred = output.data.max(1)
            correct = pred.eq(target.data).sum().item()
            accuracy = correct/input.data.size(0)   
            accuracies[split] += accuracy
            counts[split] += 1
            # Backward and optimize
            if split == "train" :
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    
    # Print info at the end of the epoch
    if accuracies["val"]/counts["val"] >= best_accuracy:
        best_accuracy = accuracies["val"]/counts["val"]
        best_epoch = epoch
        torch.save(model,'freq_best_model.pth')
    
    TrL,TrA,VL,VA= losses["train"]/counts["train"],accuracies["train"]/counts["train"],losses["val"]/counts["val"],accuracies["val"]/counts["val"]
    print("Epoch {0}: TrL={1:.4f}, TrA={2:.4f}, VL={3:.4f}, VA={4:.4f}, max VA = {5:.4f} at epoch {6:d}".format(epoch,TrL,TrA,VL,VA,
                                                                                                         best_accuracy, best_epoch))

    losses_per_epoch['train'].append(TrL)
    losses_per_epoch['val'].append(VL)
    accuracies_per_epoch['train'].append(TrA)
    accuracies_per_epoch['val'].append(VA)
