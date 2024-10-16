from collections import Counter
import math
from tqdm import tqdm
from utility.ImportanceCalculator import ImportanceCalculator


class UserUniquenessImportance(ImportanceCalculator):
    # for each hexagon we calculate the frequency of visit
    # the importance of a sequence is inverse of the sum of the frequencies of the hexagons in the sequence.
    
    def prepare(self, dataset):
        self.dataset_size = len(dataset)
        
        # set of hexagons visited by each user - user_id: set(hexagons)
        self.user_hex_set = {}
        self.user_uniqe_hex_set = {}
        self.importances = {}
        
        for sequence, uesr_id in tqdm(dataset, desc="Calculating each user hexagon set"):
            if uesr_id not in self.user_hex_set:
                self.user_hex_set[uesr_id] = set()
            self.user_hex_set[uesr_id].update(sequence)
            
        for user, hex_set in tqdm(self.user_hex_set.items(), desc="Calculating finiding the unique hexagons for each user"):
            other_users_set = set.union(*(s for u, s in self.user_hex_set.items() if u != user))
            self.importances[user] = len(hex_set - other_users_set)
            
        self.mean_importance = sum(self.importances.values()) / len(self.importances)
        self.std_importance = math.sqrt(sum((importance - self.mean_importance) ** 2 for importance in self.importances.values()) / len(self.importances))
        self.max_importance = max(self.importances.values())
        self.min_importance = min(self.importances.values())
        
    def calculate_importance(self, batch):
        sequences, user_ids = batch
        batch_importance = []
        for user in user_ids:
            batch_importance.append(self.importances[user.item()])
        
        # Min-max normalization
        batch_importance = [
            (importance - self.min_importance) / (self.max_importance - self.min_importance)
            for importance in batch_importance
        ]

        return batch_importance
