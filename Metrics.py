import torch


class Metric: # maybe consider using torch.metric
    
    def __init__(self, contMetric=False, finalMetric=False, printInfo=False) -> None:
        self.doContMetric = contMetric
        self.doFinalMetric = finalMetric
        self.printInfo = printInfo
        return None

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

    # todo accuracy, precision, recall, F1-score



class LossMetric(Metric):
    """for applying torch loss functions as metrics."""
    def __init__(self, lossFunc, contMetric=True, finalMetric=False, printInfo=False) -> None:
        super().__init__(contMetric, finalMetric, printInfo)
        self.loss = lossFunc
        self.metricName = "LossMetric"

    def getSampleMetric(self, y, y_pred):
        with torch.no_grad():
            return self.loss(y, y_pred).item()


class ConfusionMatrix():
    def __init__(self, vals) -> None:
        self.fp, self.fn, self.tp, self.tn = vals
        self.metricName = "ConfusionMatrix"

    def __repr__ (self) -> str:
        return f"<ConfusionMatrix: False postives={self.fp:.4}, False negative={self.fn:.4}, True negative={self.tn:.4}, True positive={self.tp:.4}>"

class ConfusionMetric(Metric): #confused yet?, i am.
    def __init__(self, contMetric=False, finalMetric=True, printInfo=False) -> None:
        super().__init__(contMetric, finalMetric, printInfo)
            
    def getTrainMetric(self, data):
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

