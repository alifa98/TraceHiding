from collections import Counter
import math
from scipy.stats import entropy
from tqdm import tqdm
from utility.ImportanceCalculator import ImportanceCalculator


class EntropyImportance(ImportanceCalculator):

    def prepare(self, dataset):
        self.dataset_size = len(dataset)
        self.dataset_bigram_counts = Counter()
        
        each_sequence_bigrams = []
        for sequence, uesr_id in tqdm(dataset, desc="Calculating dataset bigram counts"):
            bigrams = self.get_bigrams(sequence)
            self.dataset_bigram_counts.update(bigrams)
            
        self.total_unique_bigrams = len(self.dataset_bigram_counts)
        self.total_bigrams_count = sum(self.dataset_bigram_counts.values())
        
        self.entropies = []
        for sequence, uesr_id in tqdm(dataset, desc="Calculating dataset importance statistics"):
            probabilities = [self.dataset_bigram_counts[bigram] / self.total_bigrams_count for bigram in self.get_bigrams(sequence)]
            self.entropies.append(entropy(probabilities, base=2))
            
        self.mean_entropy = sum(self.entropies) / len(self.entropies)
        self.std_entropy = math.sqrt(sum([(entropy - self.mean_entropy) ** 2 for entropy in self.entropies]) / len(self.entropies))
        self.max_entropy = max(self.entropies)
        self.min_entropy = min(self.entropies)

    def get_bigrams(self, sequence):
        return [(sequence[i], sequence[i + 1]) for i in range(len(sequence) - 1)]

    def calculate_importance(self, batch):
        sequences, user_ids = batch
        # Calculate entropy for each sequence in the batch
        batch_entropies = []
        for seq in sequences:
            # delete padding zeros
            seq = seq[seq != 0]
            seq_bigrams = self.get_bigrams(seq.tolist()) # convert tensor to lists
            seq_probabilities = [
                self.dataset_bigram_counts[item] / self.total_bigrams_count
                for item in seq_bigrams
            ]
            seq_entropy = entropy(seq_probabilities, base=2)
            batch_entropies.append(seq_entropy)

        # Min-max normalization
        batch_entropies = [
            (entropy - self.min_entropy) / (self.max_entropy - self.min_entropy)
            for entropy in batch_entropies
        ]

        return batch_entropies
