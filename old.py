import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data


# Generate data: # maybe add to device for higher Speed
X = torch.randint(low=0, high=2, size=(100, 20), dtype=torch.float) # 100 samples of 20 ones and zeroes. 
Y = torch.randint(low=0, high=2, size=(100, 1), dtype=torch.float) # 1 for FAKE



class Model(nn.Module):
    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        self.l1 = nn.Linear(input_size, 128)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(128, 64)
        self.l3 = nn.Linear(64, output_size)
        
    def forward(self, X):
        # we have 20x1 
        out = self.l1(X)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = torch.sigmoid(self.l3(out))
        return out


loaderX = torch.utils.data.DataLoader(dataset=X, batch_size=batch_size, shuffle=False)
loaderY = torch.utils.data.DataLoader(dataset=Y, batch_size=batch_size, shuffle=False)


model = Model(input_size, output_size)

cri = nn.BCELoss() #placeholder
opt = torch.optim.Adam(model.parameters(), lr=lr) #placeholder

for epoch in range(num_epochs):
    for (x,y) in zip(loaderX,loaderY):
      # forward pass and loss
      y_pre = model(x)
      loss = cri(y_pre, y)

      # backward pass
      loss.backward()
      # update
      opt.step()
      opt.zero_grad()

    if epoch % 10 == 0:
        print(f'epoch: {epoch+1}, sample {idx}, loss = {loss.item():.4f}')