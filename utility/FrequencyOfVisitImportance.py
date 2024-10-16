from collections import Counter
import math
from tqdm import tqdm
from utility.ImportanceCalculator import ImportanceCalculator


class FrequencyImportance(ImportanceCalculator):
    # for each hexagon we calculate the frequency of visit
    # the importance of a sequence is inverse of the sum of the frequencies of the hexagons in the sequence.
    
    def prepare(self, dataset):
        self.dataset_size = len(dataset)
        self.hexagon_visit = Counter()
        
        for sequence, uesr_id in tqdm(dataset, desc="Calculating dataset visits"):
            self.hexagon_visit.update(sequence)
            
        self.total_unique_hexagons = len(self.hexagon_visit)
        self.total_hexagon_number = sum(self.hexagon_visit.values())
        
        self.importances = []
        for sequence, uesr_id in tqdm(dataset, desc="Calculating dataset importance statistics"):
            importance_inv = sum([self.hexagon_visit[hexagon] for hexagon in sequence])
            self.importances.append(1/importance_inv)
            
        self.mean_importance = sum(self.importances) / len(self.importances)
        self.std_importance = math.sqrt(sum([(importance - self.mean_importance) ** 2 for importance in self.importances]) / len(self.importances))
        self.max_importance = max(self.importances)
        self.min_importance = min(self.importances)

    def calculate_importance(self, batch):
        sequences, user_ids = batch
        # Calculate entropy for each sequence in the batch
        batch_importance = []
        for seq in sequences:
            # delete padding zeros
            seq = seq[seq != 0]
            importance_inv = sum([self.hexagon_visit[hexagon] for hexagon in seq.tolist()])
            batch_importance.append(1/importance_inv)

        # Min-max normalization
        batch_importance = [
            (importance - self.min_importance) / (self.max_importance - self.min_importance)
            for importance in batch_importance
        ]

        return batch_importance
