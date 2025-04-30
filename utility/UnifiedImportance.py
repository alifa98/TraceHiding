from utility.ImportanceCalculator import ImportanceCalculator

class UnifiedImportance(ImportanceCalculator):
    """
    Unified Importance class that combines the importance measures.
    """

    def __init__(self, importances:list[ImportanceCalculator], weights:list[float]):
        """
        Initialize the UnifiedImportance class.
        Args:
            importances (list): List of importance calculators to combine.
            weights (list): List of weights for each importance calculator.
        """
        super().__init__()
        self.importances = importances
        self.weights = weights
        
        
    def prepare(self, dataset):
        """
        Prepare the unified importance by preparing each individual importance calculator.

        Args:
            dataset (list): The dataset to prepare.
        """
        for importance in self.importances:
            importance.prepare(dataset)

    def calculate_importance(self, batch):
        """
        Calculate the unified importance by combining the importance measures.
        Args:
            batch (list): The batch of data to calculate importance for.
        Returns:
            list: The combined importance measures.
        """
        # Calculate the importance for each individual importance calculator
        importances = []
        for importance in self.importances:
            importances_list = importance.calculate_importance(batch)
            importances.append(importances_list)
            
        # Combine the importances using the weights
        combined_importance = []
        for i in range(len(importances[0])):
            combined_importance.append(sum([importances[j][i] * self.weights[j] for j in range(len(importances))]))
            
        return combined_importance