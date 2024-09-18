from tqdm import tqdm
from utility.ImportanceCalculator import ImportanceCalculator

class CoverageDiversityImportance(ImportanceCalculator):

    def prepare(self, dataset):
        self.dataset_size = len(dataset)
        self.total_unique_hexagons = set()
        for sequence, uesr_id in tqdm(dataset, desc="Calculating dataset unique hexagons"):
            self.total_unique_hexagons.update(sequence)           

    def calculate_importance(self, batch):
        sequences, user_ids = batch
        # the number of unique hexagons each sequence has
        batch_importance = []
        for seq in sequences:
            seq = seq[seq != 0]
            
            # Calculate the number of unique hexagons in the sequence
            unique_hexagons_count = len(set(seq.tolist()))
            batch_importance.append(unique_hexagons_count / len(self.total_unique_hexagons))

        return [i/max(batch_importance) for i in batch_importance]
