from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import  NearestNeighbors
import pandas as pd
from umap import UMAP

# Assuming `model` is your trained Word2Vec model
model = Word2Vec.load('PDF2JSON\combined.model')
iesg_model = Word2Vec.load('PDF2JSON\iesg.model')
sedg_model = Word2Vec.load('PDF2JSON\sedg.model')
bursa_model = Word2Vec.load('PDF2JSON\Bursa.model')
# Extract the word vectors from the model
words = list(model.wv.index_to_key)
word_vectors = np.array([model.wv[word] for word in words])



def show_tsne():
# Use t-SNE to reduce dimensions
    tsne = TSNE(n_components=2, random_state=0)
    word_vectors_2d = tsne.fit_transform(word_vectors)

    # Create and fit the model to find the nearest neighbor for each point
    neighbors = NearestNeighbors(n_neighbors=2)  # n_neighbors=2 because the closest neighbor to a point is the point itself
    neighbors.fit(word_vectors_2d)

    # Find the nearest neighbor for each point (excluding the point itself)
    distances, indices = neighbors.kneighbors(word_vectors_2d)

    # `indices` now contains the indices of the nearest neighbor for each point
    # Plotting similar to the t-SNE example
    plt.figure(figsize=(16, 16))
    # Plot points and annotations
    for i, word in enumerate(words):
        plt.scatter(word_vectors_2d[i, 0], word_vectors_2d[i, 1])
        plt.annotate(word, xy=(word_vectors_2d[i, 0], word_vectors_2d[i, 1]), 
                    xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
    # Draw lines
    for i, nearest_index in enumerate(indices[:, 1]):  # Skip the first column because it's the point itself
        start_point = word_vectors_2d[i]
        end_point = word_vectors_2d[nearest_index]
        plt.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], 'k-', alpha=0.3)  # 'k-' for black line, alpha for transparency
    plt.show()

def show_pca():
    # Use PCA to reduce dimensions
    pca = PCA(n_components=2)
    word_vectors_2d = pca.fit_transform(word_vectors)

    # Create and fit the model to find the nearest neighbor for each point
    neighbors = NearestNeighbors(n_neighbors=2)  # n_neighbors=2 because the closest neighbor to a point is the point itself
    neighbors.fit(word_vectors_2d)

    # Find the nearest neighbor for each point (excluding the point itself)
    distances, indices = neighbors.kneighbors(word_vectors_2d)

    # `indices` now contains the indices of the nearest neighbor for each point
    # Plotting similar to the t-SNE example
    plt.figure(figsize=(16, 16))
    # Plot points and annotations
    for i, word in enumerate(words):
        plt.scatter(word_vectors_2d[i, 0], word_vectors_2d[i, 1])
        plt.annotate(word, xy=(word_vectors_2d[i, 0], word_vectors_2d[i, 1]), 
                    xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
    # Draw lines
    for i, nearest_index in enumerate(indices[:, 1]):  # Skip the first column because it's the point itself
        start_point = word_vectors_2d[i]
        end_point = word_vectors_2d[nearest_index]
        plt.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], 'k-', alpha=0.3)  # 'k-' for black line, alpha for transparency
    plt.show()

def semantic_network_pca():
    common_words = set(bursa_model.wv.index_to_key) & set(iesg_model.wv.index_to_key) & set(sedg_model.wv.index_to_key)

    # Select a manageable number of words for visualization
    common_words = list(common_words)[:50]  # Adjust the number as needed
    
    # Extract vectors
    vectors_bursa = [bursa_model.wv[word] for word in common_words]
    vectors_iesg = [iesg_model.wv[word] for word in common_words]
    vectors_sedg = [sedg_model.wv[word] for word in common_words]
    all_vectors = np.array(vectors_iesg + vectors_bursa + vectors_sedg)
    pca = PCA(n_components=2)
    all_vectors_reduced = pca.fit_transform(all_vectors)
    total_words = len(common_words)
    vectors_reduced_bursa = all_vectors_reduced[:total_words]
    vectors_reduced_iesg = all_vectors_reduced[total_words:2*total_words]
    vectors_reduced_sedg = all_vectors_reduced[2*total_words:]
    plt.figure(figsize=(15, 10))

    # Scatter plot for each model
    plt.scatter(*zip(*vectors_reduced_bursa), color='red', marker='o', label='Bursa')
    plt.scatter(*zip(*vectors_reduced_iesg), color='blue', marker='x', label='iesg')
    plt.scatter(*zip(*vectors_reduced_sedg), color='green', marker='^', label='sedg')

    # Optionally, annotate some words
    for i, word in enumerate(common_words):
        plt.annotate(word, (vectors_reduced_bursa[i][0], vectors_reduced_bursa[i][1]), fontsize=8)
        plt.annotate(word, (vectors_reduced_iesg[i][0], vectors_reduced_iesg[i][1]), fontsize=8)
        plt.annotate(word, (vectors_reduced_sedg[i][0], vectors_reduced_sedg[i][1]), fontsize=8)

    plt.legend()
    plt.title("Comparison of Word Vectors from Three Different Models")
    plt.show()

def semantic_network_tsne():
    common_words = set(bursa_model.wv.index_to_key) & set(iesg_model.wv.index_to_key) & set(sedg_model.wv.index_to_key)

    # Select a manageable number of words for visualization
    common_words = list(common_words)[:100]  # Adjust the number as needed
    
    # Extract vectors
    vectors_bursa = [bursa_model.wv[word] for word in common_words]
    vectors_iesg = [iesg_model.wv[word] for word in common_words]
    vectors_sedg = [sedg_model.wv[word] for word in common_words]
    all_vectors = vectors_iesg + vectors_bursa + vectors_sedg
    all_vectors_df = pd.DataFrame(all_vectors)
    tsne = TSNE(n_components=2, random_state=0)
    all_vectors_reduced = tsne.fit_transform(all_vectors_df)
    total_words = len(common_words)
    vectors_reduced_bursa = all_vectors_reduced[:total_words]
    vectors_reduced_iesg = all_vectors_reduced[total_words:2*total_words]
    vectors_reduced_sedg = all_vectors_reduced[2*total_words:]
    plt.figure(figsize=(15, 10))

    # Scatter plot for each model
    plt.scatter(*zip(*vectors_reduced_bursa), color='red', marker='o', label='iesg')
    plt.scatter(*zip(*vectors_reduced_iesg), color='blue', marker='x', label='bursa')
    plt.scatter(*zip(*vectors_reduced_sedg), color='green', marker='^', label='sedg')

    # Optionally, annotate some words
    for i, word in enumerate(common_words):
        plt.annotate(word, (vectors_reduced_bursa[i][0], vectors_reduced_bursa[i][1]), fontsize=8)
        plt.annotate(word, (vectors_reduced_iesg[i][0], vectors_reduced_iesg[i][1]), fontsize=8)
        plt.annotate(word, (vectors_reduced_sedg[i][0], vectors_reduced_sedg[i][1]), fontsize=8)

    plt.legend()
    plt.title("Comparison of Word Vectors from Three Different Models")
    plt.show()

def semantic_network_umap():
    common_words = set(bursa_model.wv.index_to_key) & set(iesg_model.wv.index_to_key) & set(sedg_model.wv.index_to_key)

    # Select a manageable number of words for visualization
    common_words = list(common_words)[:50]  # Adjust the number as needed
    
    # Extract vectors
    vectors_bursa = [bursa_model.wv[word] for word in common_words]
    vectors_iesg = [iesg_model.wv[word] for word in common_words]
    vectors_sedg = [sedg_model.wv[word] for word in common_words]
    all_vectors = np.array(vectors_iesg + vectors_bursa + vectors_sedg)
    umap_reducer = UMAP(n_neighbors=5, min_dist=0.3, n_components=2, random_state=42)
    umap_embedding = umap_reducer.fit_transform(all_vectors)
    total_words = len(common_words)
    umap_embedding_iesg = umap_embedding[:total_words, :]
    umap_embedding_bursa = umap_embedding[total_words:2*total_words, :]
    umap_embedding_sedg = umap_embedding[2*total_words:, :]
    plt.figure(figsize=(12, 8))

    plt.scatter(umap_embedding_iesg[:, 0], umap_embedding_iesg[:, 1], color='red', label='iesg', alpha=0.6)
    plt.scatter(umap_embedding_bursa[:, 0], umap_embedding_bursa[:, 1], color='blue', label='bursa', alpha=0.6)
    plt.scatter(umap_embedding_sedg[:, 0], umap_embedding_sedg[:, 1], color='green', label='sedg', alpha=0.6)

    for i, word in enumerate(common_words):
        plt.annotate(word, (umap_embedding_bursa[i][0], umap_embedding_bursa[i][1]), fontsize=6)
        plt.annotate(word, (umap_embedding_iesg[i][0], umap_embedding_iesg[i][1]), fontsize=6)
        plt.annotate(word, (umap_embedding_sedg[i][0], umap_embedding_sedg[i][1]), fontsize=6)
    plt.legend()
    plt.title('UMAP visualization of Word Embeddings from Three Models')
    plt.show()

semantic_network_pca()

