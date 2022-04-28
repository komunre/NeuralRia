import torch
from torch import nn
import codecs
from pathlib import Path
from torch.utils.data import DataLoader
import numpy as np

from mytokenizer import WordsDataset
import mytokenizer

paths = [str(x) for x in Path("./train_data").glob("*")]

data = mytokenizer.tokenize(paths, "###---")
#data = mytokenizer.tokenize_csv("train_data/reddit.csv", 2, 2) # Shit for testing (why..?)

batch_size = 5

train_dataloader = DataLoader(WordsDataset(data, 4), batch_size=batch_size)
test_dataloader = DataLoader(WordsDataset(data, 4), batch_size=batch_size)

device = "cpu" # This is better
print(f"Using {device} device")

class NeuralNewsGen(nn.Module):
    def __init__(self, dataloader):
        super(NeuralNewsGen, self).__init__()
        self.lstm_size = 50
        self.input = nn.Embedding(
            num_embeddings=len(dataloader.dataset),
            embedding_dim=self.lstm_size
        )
        self.lstm = nn.LSTM(
            input_size=self.lstm_size,
            hidden_size=self.lstm_size,
            num_layers=3,
            dropout = 0.2,
        )
        self.stack = nn.Linear(self.lstm_size, len(dataloader.dataset))

    def forward(self, x, prev_state):
        x = self.input(x)
        output, state = self.lstm(x, prev_state)
        logits = self.stack(output)
        return logits, state
    
    def init_state(self, sequence_length):
        return (torch.zeros(3, sequence_length, self.lstm_size, device=device),
            torch.zeros(3, sequence_length, self.lstm_size, device=device))

model = NeuralNewsGen(train_dataloader).to(device)
print(model)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def train(dataloader, model, loss_fn, optimizer, epochs, seqlen=4):
    size = len(dataloader.dataset)
    model.train()

    for epoch in range(epochs):
        state_h, state_c = model.init_state(seqlen)
        for batch, (x, y) in enumerate(dataloader):
            optimizer.zero_grad()
            x = x.to(device)
            y = y.to(device)
            pred, (state_h, state_c) = model(x, (state_h, state_c))
            loss = loss_fn(pred.transpose(1, 2), y)

            state_h = state_h.detach()
            state_c = state_c.detach()

            loss.backward()
            optimizer.step()
        print(str(epoch) + "/" + str(epochs))

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            flattened = model.flatten(y)
            test_loss += loss_fn(pred, flattened).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

def simple_call(str):
        unbatched = train_dataloader.dataset.stringToIndices(str)
        batched = train_dataloader.collate_fn(unbatched).to(device)
        return train_dataloader.dataset.getByIndex(model(batched))

def predict(dataloader, model, text, next_words=9, seqlen=4):
    model.eval()

    words = text.split(' ')
    state_h, state_c = model.init_state(seqlen)
    for i in range(0, next_words):
        x = torch.tensor([[dataloader.dataset.get_word_index(w) for w in words[i:i+seqlen]]], dtype=torch.int32, device=device)
        pred, (state_h, state_c) = model(x, (state_h, state_c))
        last_word_logits = pred[0][-1]
        p = torch.nn.functional.softmax(last_word_logits, dim=0).cpu().detach().numpy()
        word_index = np.random.choice(len(last_word_logits), p=p)
        words.append(dataloader.dataset.get_word_by_index(word_index))
    
    return words

if __name__ == '__main__':
    train(train_dataloader, model, loss_fn, optimizer, 12, 4)

    print("Done!")

    torch.save(model.state_dict(), "model.pth")
    print("Saved the model to 'model.pth'!")