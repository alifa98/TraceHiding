from collections import Counter
import math
from scipy.stats import entropy
from tqdm import tqdm
from utility.ImportanceCalculator import ImportanceCalculator


class EntropyImportance(ImportanceCalculator):

    def prepare(self, dataset):
        self.dataset_size = len(dataset)

        self.dataset_bigram_counts = Counter()
        for sequence, uesr_id in tqdm(dataset, desc="Calculating dataset bigram counts"):
            self.dataset_bigram_counts.update(self.get_bigrams(sequence))

        self.total_unique_bigrams = len(self.dataset_bigram_counts)

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
            seq_bigram_counts = Counter(seq_bigrams)

            # Convert counts to probabilities
            seq_probabilities = [
                seq_bigram_counts[item] / self.dataset_bigram_counts[item]
                for item in seq_bigrams
            ]
            seq_entropy = entropy(seq_probabilities, base=2)
            batch_entropies.append(seq_entropy)

        # Normalize the entropy by dividing it by the maximum entropy
        max_entropy = math.log2(self.total_unique_bigrams)
        normalized_entropies = [entropy / max_entropy for entropy in batch_entropies]

        return normalized_entropies
