import torch
import torch.nn as nn
from EvalRecipe import *
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
    train_data=testDataSet(10, 30, device=device), # autogenerated testdata
    optimizer=torch.optim.SGD,
    loss=torch.nn.BCELoss,
    num_epochs=20,
    batch_size=2, 
    learning_rate=0.005,
    shuffleData=True
    )


train.doTrain = True
#print([p for p in train.model.parameters()])
evalR = EvalRecipe()
train.trainAllEpochs(evalR)
#print(evalR.allData)
#lossData = evalR.calculateSampleMetric(LossMetric(torch.nn.BCELoss()))

#trainData = evalR.calculateTrainMetrics(ConfusionMetric(), PrecisionMetric(), RecallMetric(), HarmonicMeanMetric())
trainData = evalR.calculateEpochMetrics(20, ConfusionMetric(), PrecisionMetric(), RecallMetric(), HarmonicMeanMetric())

#trainData = evalR.calculateTrainMetrics(ConfusionMetric())
print(trainData)



# plot
#fig, ax = plt.subplots()

#ax.plot([i+1 for i in range(len(lossData[0]))],lossData[0])

#ax.set(xlim=(0, 100), xticks=np.arange(0, 101, 5),
       #ylim=(0, 1), yticks=np.arange(0, 1,0.1))

#plt.show()

