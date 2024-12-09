from src.agent.rl_agent import BenchmarkTransformer, AttentionMLP

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
import os
import datetime

df = pd.read_csv('data/demos/training_demo.csv')
# Uncomment this after running the process_data.py script
# df = pd.read_csv('data/processed/time_series_2024-09-25_26XYeULTEduepz4qQd6EvqcLsLnmLQ7pEUhxHPKSoru5.csv')

df['historical_price'] = df['historical_price'].apply(lambda x: np.array(eval(x), dtype=np.float32))

X = df.drop(columns=['pair_address', 'label', 'current_price'])
y = df['label'].values

X = np.stack(X['historical_price'].values)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = torch.tensor(X_train[:, :, None], dtype=torch.float32)
X_test = torch.tensor(X_test[:, :, None], dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

exp_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = f"logs/{exp_name}"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

writer = SummaryWriter(log_dir=log_dir)

# Hyperparameters
input_dim = X_train.shape[2]
nhead = 4
num_layers = 2
dim_feedforward = 64
dropout = 0.1
learning_rate = 0.001
num_epochs = 10

# model = BenchmarkTransformer(input_dim, nhead, num_layers, dim_feedforward, dropout)
model = AttentionMLP(input_dim, nhead, num_layers, dim_feedforward, dropout)


criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

print(f"Training Start")
print(f"Model: {model}")
print(f"Loss Function: {criterion}")
print(f"Optimizer: {optimizer}")


for epoch in range(num_epochs):
    model.train()

    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    writer.add_scalar('Loss/train', loss.item(), epoch)

    if (epoch + 1) % 2 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

model.eval()
with torch.no_grad():
    test_outputs = model(X_test)
    test_loss = criterion(test_outputs, y_test)
    print(f'Test Loss: {test_loss.item():.4f}')

    writer.add_scalar('Loss/test', test_loss.item(), num_epochs)

writer.close()
