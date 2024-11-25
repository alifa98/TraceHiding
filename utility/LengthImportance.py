from tqdm import tqdm
from utility.ImportanceCalculator import ImportanceCalculator


class TrajectoryLengthImportance(ImportanceCalculator):

    def prepare(self, dataset):
        self.dataset_size = len(dataset)
        self.lens = []
        for sequence, user_id in tqdm(dataset, desc="Calculating tajectory lengths"):
            self.lens.append(len(sequence))
            
        self.max_length = max(self.lens)
        self.min_length = min(self.lens)
        
    def calculate_importance(self, batch):
        sequences, user_ids = batch
        batch_lengths = []
        for seq in sequences:
            # delete padding zeros
            seq = seq[seq != 0]
            batch_lengths.append(len(seq))

        # Min-max normalization
        batch_lengths = [
            (item - self.min_length) / (self.max_length - self.min_length)
            for item in batch_lengths
        ]

        return batch_lengths
