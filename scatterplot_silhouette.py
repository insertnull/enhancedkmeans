import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import calinski_harabasz_score, pairwise_distances, silhouette_score

# Load datasets
movies = pd.read_csv('ml-latest-small/movies.csv')
ratings = pd.read_csv('ml-latest-small/ratings.csv')

# Extract genres
genres = ['Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 
          'Documentary', 'Drama', 'Fantasy', 'Horror', 'Mystery', 'Romance', 
          'Sci-Fi', 'Thriller', 'War', 'Western']

# Assign binary genre flags
for genre in genres:
    movies[genre] = movies['genres'].str.contains(genre, regex=False).astype(int)

# Merge ratings with movies
df = ratings.merge(movies, on='movieId')

# Compute user average ratings per genre
user_genre_ratings = df.groupby(['userId'])[genres].mean()

# Make a copy before modifying (Fixes KeyError)
user_genre_ratings_original = user_genre_ratings.copy()

# Standardize data for clustering
scaler = StandardScaler()
user_genre_ratings_scaled = scaler.fit_transform(user_genre_ratings)

### --- ORIGINAL K-MEANS (EUCLIDEAN DISTANCE) ---
kmeans_original = KMeans(n_clusters=2, random_state=42, n_init=10)
clusters_original = kmeans_original.fit_predict(user_genre_ratings_scaled)

# Store clusters back in the original DataFrame (Fix)
user_genre_ratings_original["Cluster_Original"] = clusters_original

### --- LOF OUTLIER REMOVAL ---
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.5, metric="canberra")  # Stricter settings
outliers_lof = lof.fit_predict(user_genre_ratings_scaled)

# Keep only inliers
user_genre_ratings_filtered = user_genre_ratings[outliers_lof == 1]
user_genre_ratings_filtered_scaled = scaler.fit_transform(user_genre_ratings_filtered)

### --- CUSTOM K-MEANS USING CANBERRA DISTANCE ---
def custom_kmeans_canberra(X, n_clusters, max_iter=300, random_state=42):
    np.random.seed(random_state)
    initial_indices = np.random.choice(X.shape[0], n_clusters, replace=False)
    centers = X[initial_indices]

    for _ in range(max_iter):
        distances = pairwise_distances(X, centers, metric='canberra')
        labels = np.argmin(distances, axis=1)
        new_centers = []
        for i in range(n_clusters):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                new_centers.append(cluster_points.mean(axis=0))
            else:
                new_centers.append(centers[i])  # Keep old center if cluster is empty
        new_centers = np.array(new_centers)
        if np.allclose(centers, new_centers, atol=1e-4):
            break
        centers = new_centers

    return labels, centers

# Determine optimal number of clusters using Calinski-Harabasz Index
best_k = None
best_score = -np.inf

for k in range(2, 6):  # Testing different k values
    labels, _ = custom_kmeans_canberra(user_genre_ratings_filtered_scaled, n_clusters=k)
    score = calinski_harabasz_score(user_genre_ratings_filtered_scaled, labels)
    
    if score > best_score:
        best_k = k
        best_score = score

# Apply Enhanced K-Means with Canberra Distance
clusters_enhanced, _ = custom_kmeans_canberra(user_genre_ratings_filtered_scaled, n_clusters=best_k)

# Store clusters back in the filtered DataFrame
user_genre_ratings_filtered["Cluster_Enhanced"] = clusters_enhanced

### --- PLOT SIDE-BY-SIDE SCATTERPLOTS + SILHOUETTE SCORE ---
def plot_side_by_side(genre_x, genre_y):
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))  # 1 row, 2 columns for side-by-side comparison

    # Original K-Means Scatterplot
    ax = axes[0]
    if "Cluster_Original" in user_genre_ratings_original.columns:
        for cluster in sorted(user_genre_ratings_original["Cluster_Original"].unique()):
            subset = user_genre_ratings_original[user_genre_ratings_original["Cluster_Original"] == cluster]
            ax.scatter(subset[genre_x], subset[genre_y], label=f'Cluster {cluster}', alpha=0.7)
    ax.set_xlabel(f'Average {genre_x} Rating')
    ax.set_ylabel(f'Average {genre_y} Rating')
    ax.set_title(f'Original K-Means (Euclidean) - {genre_x} vs {genre_y}')
    ax.legend()

    # Enhanced K-Means Scatterplot
    ax = axes[1]
    if "Cluster_Enhanced" in user_genre_ratings_filtered.columns:
        for cluster in sorted(user_genre_ratings_filtered["Cluster_Enhanced"].unique()):
            subset = user_genre_ratings_filtered[user_genre_ratings_filtered["Cluster_Enhanced"] == cluster]
            ax.scatter(subset[genre_x], subset[genre_y], label=f'Cluster {cluster}', alpha=0.7)
    ax.set_xlabel(f'Average {genre_x} Rating')
    ax.set_ylabel(f'Average {genre_y} Rating')
    ax.set_title(f'Enhanced K-Means (Canberra, LOF, Optimal k) - {genre_x} vs {genre_y}')
    ax.legend()

    plt.show()

### --- Compute Silhouette Scores for Genre Pairs ---
def compute_silhouette(genre_x, genre_y):
    # Select only the genre pair columns
    original_data = user_genre_ratings_original[[genre_x, genre_y]].values
    enhanced_data = user_genre_ratings_filtered[[genre_x, genre_y]].values

    # Compute silhouette scores
    silhouette_original = silhouette_score(original_data, user_genre_ratings_original["Cluster_Original"])
    silhouette_enhanced = silhouette_score(enhanced_data, user_genre_ratings_filtered["Cluster_Enhanced"])

    print(f"Silhouette Score for {genre_x} vs {genre_y}:")
    print(f"  Original K-Means: {silhouette_original:.4f}")
    print(f"  Enhanced K-Means (Canberra + LOF + Optimal k): {silhouette_enhanced:.4f}\n")

# Genre pairs to evaluate
genre_pairs = [('Romance', 'Drama'), ('Sci-Fi', 'Fantasy'), ('Action', 'Adventure')]

for genre_x, genre_y in genre_pairs:
    compute_silhouette(genre_x, genre_y)  # Compute silhouette score
    plot_side_by_side(genre_x, genre_y)   # Plot side-by-side scatterplots
