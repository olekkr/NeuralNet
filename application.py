import torch
import torch.nn as nn
from NeuralNet import * 
from Metrics import * 
import matplotlib.pyplot as plt
import numpy as np

if torch.cuda.is_available():
    device = torch.device("cuda")

modelElems = [
    nn.Linear(30,64, device=device),  
    nn.ReLU(),
    nn.Linear(64,1, device=device),
    nn.Sigmoid()
    ]
model = Model(modelElems, device=device)


#print(model.forward(torch.randint(low=0, high=2, size=(5,20), dtype=torch.float)))


train = TrainHandler(
    model=model, 
    train_data=testDataSet(1, 30, device=device), # autogenerated testdata
    optimizer=torch.optim.SGD,
    loss=torch.nn.BCELoss,
    num_epochs=100,
    batch_size=2, 
    learning_rate=0.001,
    shuffleData=True
    )

#train.trainWithPlot(metrics=[LossMetric(torch.nn.BCELoss())])
finData, contData = train.train(metrics=[LossMetric(torch.nn.BCELoss())])
print(finData)

# plot
fig, ax = plt.subplots()

ax.plot([i+1 for i in range(len(contData))],contData)

ax.set(xlim=(0, 101), xticks=np.arange(0, 101,5),
       ylim=(0, 101), yticks=np.arange(0, 101,5))

plt.show()


#print([p for p in model.parameters()])
#print(torch.optim.SGD(model.parameters(),lr=0.0001))

evaluator = EvalHandler(
    model=model,
    val_data=testDataSet(1, 30, device=device)
    )
#conf = 
a = evaluator.evaluate(ConfusionMetric())

print(f"{a}")