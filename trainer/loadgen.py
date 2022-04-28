from itertools import count
import wordgen2
from torch.utils.data import DataLoader
from pathlib import Path
from torch.utils.data import DataLoader
import torch
from mytokenizer import WordsDataset
import mytokenizer


model = wordgen2.NeuralNewsGen(wordgen2.train_dataloader).to(device=wordgen2.device)
model.load_state_dict(torch.load("model.pth"))

train_dataloader = wordgen2.train_dataloader

print("Vocab size: " + str(len(train_dataloader.dataset)))

model.eval()

heh = input("Enter begin string: ")
counter = int(input("Enter words number: "))

print(' '.join(wordgen2.predict(wordgen2.train_dataloader, model, heh, counter)))