import os
import matplotlib.pyplot as plt
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

DATASET_NAME = "HO_Porto_Res8"

train_dataset = torch.load(f"experiments/{DATASET_NAME}/splits/{DATASET_NAME}_train.pt", weights_only=False)
test_dataset = torch.load(f"experiments/{DATASET_NAME}/splits/{DATASET_NAME}_test.pt", weights_only=False)

os.makedirs(f"analysis/{DATASET_NAME}/plots", exist_ok=True)

def analyze_and_visualize(dataset, dataset_name="Dataset"):
   print(f"Analyzing {dataset_name}...")
   
   # Clustering of Sequences
   sequences_str = [' '.join(map(str, seq)) for seq, label in dataset]
   vectorizer = TfidfVectorizer()
   X = vectorizer.fit_transform(sequences_str)

   ## Color by K-Means Clustering
   # n_clusters = 5
   # kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
   # clusters = kmeans.fit_predict(X)
   
   ## Color by labels
   labels = [label for _, label in dataset]
   clusters = labels

   pca = PCA(n_components=2)
   X_pca = pca.fit_transform(X.toarray())

   plt.figure(figsize=(10, 6))
   plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', s=50, alpha=0.7)
   plt.title(f'KMeans Clustering of Sequences in {dataset_name}')
   plt.xlabel('PCA Component 1')
   plt.ylabel('PCA Component 2')
   plt.colorbar(label='Cluster')
   
   # Save the plot
   plt.savefig(f"analysis/{DATASET_NAME}/plots/{dataset_name}_clustering.png")


analyze_and_visualize(train_dataset, "Train Dataset")
analyze_and_visualize(test_dataset, "Test Dataset")