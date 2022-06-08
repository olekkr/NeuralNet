import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Dataset



def testDataSet(sampleCount, featureCount,device="cpu"):
    '''
    generates a random 
    '''
    X = torch.randint(low=0, high=2, size=(sampleCount, featureCount), dtype=torch.float, device=device) # 100 samples of 20 ones and zeroes. 
    Y = torch.randint(low=0, high=2, size=(sampleCount, 1), dtype=torch.float, device=device) # 1 for FAKE
    my_dataset = TensorDataset(X,Y)
    return my_dataset


class Model(nn.Module):
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
    def __init__(self, model, train_data:Dataset, optimizer:torch.optim, loss:torch.nn, shuffleData, batch_size, num_epochs, learning_rate) -> None:
        # model to train 
        self.model = model
        
        # creates a dataloader from dataset which will be used for training
        self.loader = DataLoader(dataset=train_data,shuffle=shuffleData, batch_size=batch_size)

        #
        self.optimizer = optimizer(model.parameters(), learning_rate)
        self.loss = loss()
        self.num_epochs = num_epochs

    def trainSingleEpoch(self):
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
        return (loss.item())

    def train(self ):
        data = []
        # does all epochs
        for e in range(self.num_epochs):
            loss = self.trainSingleEpoch()
            data += [e+1, loss]
        return data

class Metric: # maybe consider using torch.metric
    """
    Abstract class 
    takes care of evaluation
    """

    def calcMetric(data): # (y,y_pred)
        return None

    # todo accuracy, precision, recall, F1-score


        
class ConfusionMatrix():
    def __init__(self, vals) -> None:
        self.fp, self.fn, self.tp, self.tn = vals

    def __repr__ (self) -> str:
        return f"<ConfusionMatrix: False postives={self.fp}, False negative{self.fn}, True negative{self.tn}, True positive={self.tp}>"

class ConfusionMetric(Metric): #confused yet?, i am.
    def __init__(self) -> None:
        super().__init__()
    
    def calcMetric(self, data):
        size = len(data)
        fp, fn, tp, tn = 0, 0, 0, 0
        for (y,y_pred) in data:
            if y == round(y_pred): # true
                if y == 0: # true negative
                    tn+=1 
                    continue
                else: # true positive
                    tp+=1
                    continue
            else: # false
                if y == 0: # false negative
                    fn+=1 
                    continue
                else: # false positive
                    fp+=1
                    continue
        
        return ConfusionMatrix([fp/size, fn/size, tp/size, tn/size])
    


class EvalHandler:
    def __init__(self, model, val_data:Dataset) -> None:
        # model to train 
        self.model = model
        
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
        return [metric.calcMetric(data) for metric in metrics]

