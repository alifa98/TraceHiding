# Class for Importance calculating
class ImportanceCalculator(object):
    
    def prepare(self, dataset):
        raise NotImplementedError
    
    def calculate_importance(self, batch):
        raise NotImplementedError