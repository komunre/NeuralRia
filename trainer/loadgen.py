import wordgen2
from torch.utils.data import DataLoader
from pathlib import Path
from torch.utils.data import DataLoader
import torch
from mytokenizer import WordsDataset
import mytokenizer


model = wordgen2.NeuralNewsGen()
model.load_state_dict(torch.load("model.pth"))

train_dataloader = wordgen2.train_dataloader

model.eval()

heh = input("Enter begin string: ")

def simple_call(str):
    unbatched = train_dataloader.dataset.stringToIndices(str)
    batched = train_dataloader.collate_fn(unbatched)
    return train_dataloader.dataset.getByIndex(model(batched))

heh = heh + simple_call(heh)

generated = heh
for i in range(10):
    addition = simple_call(generated)
    generated = generated + ' ' + addition

print(generated)