import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import calinski_harabasz_score, silhouette_score

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

# Standardize data for clustering
scaler = StandardScaler()
user_genre_ratings_scaled = scaler.fit_transform(user_genre_ratings)

### --- ORIGINAL K-MEANS (EUCLIDEAN DISTANCE) ---
kmeans_original = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters_original = kmeans_original.fit_predict(user_genre_ratings_scaled)
user_genre_ratings["Cluster_Original"] = clusters_original

### --- ENHANCED K-MEANS (USING CALINSKI-HARABASZ INDEX) ---
best_k = None
best_score = -np.inf

for k in range(2, 6):  # Testing different k values
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(user_genre_ratings_scaled)
    score = calinski_harabasz_score(user_genre_ratings_scaled, labels)
    
    if score > best_score:
        best_k = k
        best_score = score

# Apply Enhanced K-Means with best k
kmeans_enhanced = KMeans(n_clusters=best_k, random_state=42, n_init=10)
clusters_enhanced = kmeans_enhanced.fit_predict(user_genre_ratings_scaled)
user_genre_ratings["Cluster_Enhanced"] = clusters_enhanced

### --- PLOT SIDE-BY-SIDE SCATTERPLOTS + SILHOUETTE SCORE ---
def plot_side_by_side(genre_x, genre_y):
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    # Original K-Means Scatterplot
    ax = axes[0]
    for cluster in sorted(user_genre_ratings["Cluster_Original"].unique()):
        subset = user_genre_ratings[user_genre_ratings["Cluster_Original"] == cluster]
        ax.scatter(subset[genre_x], subset[genre_y], label=f'Cluster {cluster}', alpha=0.7)
    ax.set_xlabel(f'Average {genre_x} Rating')
    ax.set_ylabel(f'Average {genre_y} Rating')
    ax.set_title(f'Original K-Means - {genre_x} vs {genre_y}')
    ax.legend()

    # Enhanced K-Means Scatterplot
    ax = axes[1]
    for cluster in sorted(user_genre_ratings["Cluster_Enhanced"].unique()):
        subset = user_genre_ratings[user_genre_ratings["Cluster_Enhanced"] == cluster]
        ax.scatter(subset[genre_x], subset[genre_y], label=f'Cluster {cluster}', alpha=0.7)
    ax.set_xlabel(f'Average {genre_x} Rating')
    ax.set_ylabel(f'Average {genre_y} Rating')
    ax.set_title(f'Enhanced K-Means (Calinski-Harabasz Optimal k) - {genre_x} vs {genre_y}')
    ax.legend()

    plt.show()

### --- Compute Silhouette Scores for Genre Pairs ---
def compute_silhouette(genre_x, genre_y):
    original_data = user_genre_ratings[[genre_x, genre_y]].values
    enhanced_data = user_genre_ratings[[genre_x, genre_y]].values

    silhouette_original = silhouette_score(original_data, user_genre_ratings["Cluster_Original"])
    silhouette_enhanced = silhouette_score(enhanced_data, user_genre_ratings["Cluster_Enhanced"])

    print(f"Silhouette Score for {genre_x} vs {genre_y}:")
    print(f"  Original K-Means: {silhouette_original:.4f}")
    print(f"  Enhanced K-Means (Calinski-Harabasz Optimal k): {silhouette_enhanced:.4f}\n")

# Genre pairs to evaluate
genre_pairs = [('Romance', 'Drama'), ('Sci-Fi', 'Fantasy'), ('Action', 'Adventure')]

for genre_x, genre_y in genre_pairs:
    compute_silhouette(genre_x, genre_y)
    plot_side_by_side(genre_x, genre_y)
