# Class for Importance calculating
class ImportanceCalculator(object):
    def __init__(self, data_format=None):
        self.data_format = data_format
        
    def prepare(self, dataset):
        raise NotImplementedError
    
    def calculate_importance(self, batch):
        raise NotImplementedError