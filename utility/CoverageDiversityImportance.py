import math
from tqdm import tqdm
from utility.ImportanceCalculator import ImportanceCalculator

class CoverageDiversityImportance(ImportanceCalculator):

    def prepare(self, dataset):
        self.dataset_size = len(dataset)
        self.total_unique_hexagons = set()
        
        self.sequnce_diversity = []
        
        if self.data_format == "transformer":
            #the data is for transformer models thus the input is list of dicts
            dataset = [(item["input_ids"], item["labels"]) for item in dataset]
        
        for sequence, uesr_id in tqdm(dataset, desc="Calculating dataset unique hexagons"):
            self.total_unique_hexagons.update(sequence)
            
        for sequence, uesr_id in tqdm(dataset, desc="Calculating dataset sequence diversity"):
            self.sequnce_diversity.append(len(set(sequence))/len(self.total_unique_hexagons))
            
        self.mean_diversity = sum(self.sequnce_diversity) / len(self.sequnce_diversity)
        self.std_diversity = math.sqrt(sum([(diversity - self.mean_diversity) ** 2 for diversity in self.sequnce_diversity]) / len(self.sequnce_diversity))
        self.max_diversity = max(self.sequnce_diversity)
        self.min_diversity = min(self.sequnce_diversity)
        

    def calculate_importance(self, batch):
        
        if self.data_format == "transformer":
            #the data is for transformer models thus the input is list of dicts
            sequences = batch["input_ids"]
        else:
            #the data is for non-transformer models thus the input is list of tuples
            sequences, user_ids = batch
            
        # the number of unique hexagons each sequence has
        batch_importance = []
        for seq in sequences:
            seq = seq[seq != 0]
            
            # Calculate the number of unique hexagons in the sequence
            unique_hexagons_count = len(set(seq.tolist()))
            batch_importance.append(unique_hexagons_count / len(self.total_unique_hexagons))
            
        # Min-max normalization
        batch_importance = [
            (importance - self.min_diversity) / (self.max_diversity - self.min_diversity)
            for importance in batch_importance
        ]

        return batch_importance