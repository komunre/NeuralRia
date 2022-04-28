import torch
from torch import nn
import codecs
from pathlib import Path
from torch.utils.data import DataLoader

from mytokenizer import WordsDataset
import mytokenizer

paths = [str(x) for x in Path("./train_data").glob("*")]

data = mytokenizer.tokenize(paths, "###---")

batch_size = 5

train_dataloader = DataLoader(WordsDataset(data), batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(WordsDataset(data), batch_size=batch_size)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

class NeuralNewsGen(nn.Module):
    def __init__(self):
        super(NeuralNewsGen, self).__init__()
        self.input = nn.Flatten()
        self.stack = nn.Linear(5, 5)

    def forward(self, x):
        x = self.input(x)
        logits = self.stack(x)
        return logits

model = NeuralNewsGen().to(device)
print(model)

loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (x, y) in enumerate(dataloader):
        x = x.to(device)
        pred = model(x)
        flattened = model.input(y)
        loss = loss_fn(pred, flattened)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(x)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            flattened = model.input(y)
            test_loss += loss_fn(pred, flattened).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")



if __name__ == '__main__':
    epochs = 25
    for t in range(epochs):
        train(train_dataloader, model, loss_fn, optimizer)
        test(test_dataloader, model, loss_fn)

    print("Done!")

    torch.save(model.state_dict(), "model.pth")
    print("Saved the model to 'model.pth'!")

    heh = "Путин назвал Украину холодной США "

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