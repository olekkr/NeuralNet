from NeuralNet import * 
from Metrics import * 

class EvalRecipe:
    def __init__(self, printProgress=False) -> None:
        self.printProgress = printProgress

        self.doCollectSampleMetrics = False
        self.doTrainMetrics = False

        self.allData = []
        '''
        self.datasetSize = datasetSize
        self.epochs = epochs
        self.batchSize = batchSize
        '''
            
            
    def collectSampleMetrics(self, y_batch, y_pred_batch):
        for y, y_pred in zip(y_batch, y_pred_batch):
            self.allData.append([y,y_pred])

            if self.doCollectSampleMetrics:
                for idx, sampleMetric in enumerate(self.sampleMetrics):
                    # add metricData
                    metricValue = sampleMetric.getSampleMetric(y,y_pred)
                    self.sampleMetricData[idx].append(metricValue)

    def calculateSampleMetric (self, *sampleMetrics):
        sampleMetricsData = [[] for _ in sampleMetrics]
        for metricNum, sampleMetric in enumerate(sampleMetrics):
            for y, y_pred in self.allData:
                sampleMetricsData[metricNum].append(sampleMetric.getSampleMetric(y, y_pred))
        return sampleMetricsData 

    def calculateTrainMetrics (self, *trainMetrics):
        print(self.allData)
        trainMetricsData = []
        for trainMetric in trainMetrics:
            trainMetricsData += trainMetric.getTrainMetric(self.allData)
        return trainMetricsData
                

           
    
