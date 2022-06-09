from cmath import nan
import torch


class Metric: # maybe consider using torch.metric
    """
    Abstract class 
    takes care of evaluation
    """
    def getSampleMetric(self, data): # (y,y_pred) as TENSORS
        '''
        getSampleMetric is a metric which returns a value each sample
        '''
        raise NotImplementedError

    def getEpochMetric(self, data): # (y,y_pred) as TENSORS
        '''
        getEpochMetric is a metric which returns a value each epoch
        '''
        raise NotImplementedError
    
    def getTrainMetric(self, data): # (y,y_pred) as TENSORS
        '''
        getTrainMetric is a metric that is returned continously over all samples during training.
        '''
        raise NotImplementedError

    def printMaybe(self, epoch, y, y_pred):
        with torch.no_grad(): 
            if self.printInfo:
                print(f"Epoch {epoch} : {self.metricName}: {self.getContniousMetric(y, y_pred)}")
    # todo accuracy, precision, recall, F1-score

def confusionMatrix(data):
    size = len(data)
    fp, fn, tp, tn = 0, 0, 0, 0
    for (_y,_y_pred) in data:
        y = _y.item()
        y_pred = _y_pred.item()
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
    try:
        return fp/size, fn/size, tp/size, tn/size
    except ZeroDivisionError:
        return 0, 0, 0, 0 

class LossMetric(Metric):
    """for applying torch loss functions as metrics."""
    def __init__(self, lossFunc) -> None:
        self.loss = lossFunc

    def getSampleMetric(self, y, y_pred):
        with torch.no_grad():
            return self.loss(y, y_pred).item()


class ConfusionMetric(Metric):
    def getTrainMetric(self, data):
        return confusionMatrix(data)

class AccuracyMetric(Metric):
    def getTrainMetric(self, data):
        fp, fn, tp, tn = confusionMatrix(data)
        return tp + tn

class PrecisionMetric(Metric):
    def getTrainMetric(self, data):
        fp, fn, tp, tn = confusionMatrix(data)
        try:
            result = tp/(tp+fp)
        except ZeroDivisionError:
            result = nan
        return result

class RecallMetric(Metric):
    def getTrainMetric(self, data):
        fp, fn, tp, tn = confusionMatrix(data)
        try:
            result = tp/(tp+fn)
        except ZeroDivisionError:
            result = nan
        return result

class HarmonicMeanMetric(Metric):
    def getTrainMetric(self, data):
        fp, fn, tp, tn = confusionMatrix(data)
        try:
            recall = tp/(tp+fn)
            precision = tp/(tp+fp)
            result = (2*precision*recall)/(precision+recall)
        except ZeroDivisionError:
            result = nan
        return result