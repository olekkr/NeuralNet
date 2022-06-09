import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Dataset
import matplotlib as plt
from Metrics import *


def testDataSet(sampleCount, featureCount, device="cpu"):
    '''Generates a random dataset'''
    X = torch.randint(low=0, high=2, size=(sampleCount, featureCount), dtype=torch.float, device=device) # 100 samples of 20 ones and zeroes. 
    Y = torch.randint(low=0, high=2, size=(sampleCount, 1), dtype=torch.float, device=device) # 1 for FAKE
    my_dataset = TensorDataset(X,Y)
    return my_dataset


class Model(nn.Module):
    """Encapsulates a neural network"""
    def __init__(self, params, device = "cpu") -> None:
        super(Model,self).__init__()
        self.device = device
        self.params = nn.ModuleList(params).to(device) # required for param-discovery for optim.
    
    def forward(self, x):
        """
        does a single forward-pass 
        """
        out = self.params[0](x)
        for e in self.params[1:]:
            out = e(out)
        return out
    
    def debug(self):
        print(self.params)


class TrainHandler():
    """Handles the training of a neural network"""
    def __init__(self, model, train_data:Dataset, optimizer:torch.optim, loss:torch.nn, shuffleData, batch_size, num_epochs, learning_rate) -> None:
        # model to train 
        self.model = model
        
        # creates a dataloader from dataset which will be used for training
        self.loader = DataLoader(dataset=train_data,shuffle=shuffleData, batch_size=batch_size)

        # Initializes the optimizer 
        self.optimizer = optimizer(model.parameters(), learning_rate)
        self.loss = loss()
        self.num_epochs = num_epochs

    def train(self, metrics=[]):
        """Trains the neural network """
        
        finData = [] # data for final Metrics
        contData = [] # data *from* final Metrics

        print(metrics)
        anyContData = any(metric.doContMetric for metric in metrics)
        anyFinData = any(metric.doFinalMetric for metric in metrics)

        # does all epochs
        for e in range(self.num_epochs):
            for i, (x, y) in enumerate(self.loader):
                # send to device
                y = y.to(self.model.device)
                x = x.to(self.model.device)

                # forward pass
                y_pred = self.model.forward(x)
                #calculate loss
                loss = self.loss(y_pred,y)

                # zero all paramgradients
                self.optimizer.zero_grad()
                # calculate gradients
                loss.backward()
                # update parameters based on optimizer
                self.optimizer.step()

                # collects data for finalMetrics
                if anyFinData:
                    finData.append([y.item(), y_pred.item()])
                # calculates data from Continous metrics
                if anyContData:
                    contData.append([metric.getContniousMetric(y, y_pred) for metric in metrics if metric.doContMetric])
            
            # 
            if e+1 % 100 == 0:
                for m in metrics:
                    m.printMaybe(e+1, y, y_pred)  

        finalMetricData = [metric.getFinalMetric(finData) for metric in metrics if metric.doFinalMetric]

        return [finalMetricData, contData]
    
    def trainWithPlot(self, axis=None, metrics=[], file=None):
        finalMetricData, contData = self.train(metrics)
        print(finalMetricData)
        print(contData)





class EvalHandler:
    """Handles evaluation of a neural net"""
    def __init__(self, model, val_data:Dataset, loss=torch.nn.BCELoss) -> None:
        # model to train 
        self.model = model
        self.loss = loss()
        # creates a dataloader from dataset which will be used for training
        self.loader = DataLoader(dataset=val_data, batch_size=1)

    def evaluate(self, *metrics):
        data = []
        for i, (x, y) in enumerate(self.loader):
            with torch.no_grad(): # dont update gradients during evaluation
                # send to device
                y = y.to(self.model.device)
                x = x.to(self.model.device)

                # forward pass
                y_pred = self.model.forward(x)
                
                data.append([y.item(), y_pred.item()])
        finalMetricData = [metric.getFinalMetric(data) for metric in metrics if metric.doFinalMetric]
        return finalMetricData
    
    def evaluateWithPlot(self, file, axis, *metrics):
        data = self.evaluate(metrics) 

