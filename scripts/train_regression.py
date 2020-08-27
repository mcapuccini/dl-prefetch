# %%
# Imports
import torch
from torch import nn, optim
from helpers import train_loop
from matplotlib import pyplot as plt

# %%
# Define model architecture
class Model(nn.Module):
  def __init__(self):
    super(Model, self).__init__()
    self.lstm = nn.LSTM(
        input_size=1,
        hidden_size=50,
        batch_first=True)
    self.dropout = nn.Dropout(p=0.1)
    self.linear = nn.Linear(
      in_features=50,
      out_features=1
    )
  def forward(self, x):
    out, _ = self.lstm(x) # h0, c0 default to 0
    out = self.dropout(out[:, -1, :]) # [:, -1, :] slices out the last hidden state
    out = self.linear(out)
    return out

# %%
# Load data
train = torch.load('../data/canneal_test/deltas_ord_norm.train.64.pt')
dev = torch.load('../data/canneal_test/deltas_ord_norm.dev.64.pt')

# %%
# Train
model = Model()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters())
_, tr_history, dev_history = train_loop(
  train_x=train[:,:-1].unsqueeze(2),
  train_y=train[:,-1].reshape(-1,1),
  dev_x=dev[:,:-1].unsqueeze(2),
  dev_y=dev[:,-1].reshape(-1,1),
  model=model,
  criterion=criterion,
  optimizer=optimizer,
  n_epochs=20,
  batch_size=256,
)