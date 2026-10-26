import torch
import torch.nn as nn
import pandas as pd
import torch.nn.functional as f
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class model(nn.Module):
    def __init__(self, il, hl1, ol):
        super().__init__()
        self.fc1 = nn.Linear(il, hl1)
        self.fc2 = nn.Linear(hl1, ol)

    def forward(self, x):
        x = f.relu(self.fc1(x))
        x = self.fc2(x)

        return x

seed = 41
torch.manual_seed(seed)

model = model(5, 8, 2)

url = 'cat_dog_numerical_dataset.csv'
data = pd.read_csv(url)

X = data.drop('output', axis=1).values
y = data['output'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)


cri = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr=0.01)

losses = []

epochs = 100

for epoch in range(epochs):
    y_pred = model(X_train)
    loss = cri(y_pred, y_train)

    opt.zero_grad()
    loss.backward()
    opt.step()

    losses.append(loss.detach().item())

    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

    
# Evaluate the model
model.eval()

with torch.no_grad():
    y_eval = model(X_test)
    y_predicted = torch.argmax(y_eval, dim=1)
    accuracy = (y_predicted == y_test).float().mean()

print("Test accuracy:", round(accuracy.item() * 100, 2), "%")