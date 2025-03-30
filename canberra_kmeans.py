import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, pairwise_distances

# Load datasets
movies = pd.read_csv('ml-latest-small/movies.csv')
ratings = pd.read_csv('ml-latest-small/ratings.csv')

# Define genres and assign binary flags
genres = ['Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 
          'Documentary', 'Drama', 'Fantasy', 'Horror', 'Mystery', 'Romance', 
          'Sci-Fi', 'Thriller', 'War', 'Western']

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

# Compute silhouette score for Euclidean-based clustering
silhouette_original = silhouette_score(user_genre_ratings_scaled, clusters_original, metric='euclidean')

### --- CUSTOM K-MEANS USING CANBERRA DISTANCE ---
def custom_kmeans_canberra(X, n_clusters, max_iter=300, random_state=42):
    np.random.seed(random_state)
    initial_indices = np.random.choice(X.shape[0], n_clusters, replace=False)
    centers = X[initial_indices]

    for _ in range(max_iter):
        distances = pairwise_distances(X, centers, metric='canberra')
        labels = np.argmin(distances, axis=1)
        new_centers = np.array([X[labels == i].mean(axis=0) for i in range(n_clusters)])
        if np.allclose(centers, new_centers, atol=1e-4):
            break
        centers = new_centers

    return labels, centers

# Apply Enhanced K-Means with Canberra Distance
chosen_k = 3
clusters_enhanced, _ = custom_kmeans_canberra(user_genre_ratings_scaled, n_clusters=chosen_k)
user_genre_ratings["Cluster_Enhanced"] = clusters_enhanced

# Compute silhouette score for Canberra-based clustering
silhouette_enhanced = silhouette_score(user_genre_ratings_scaled, clusters_enhanced, metric='canberra')

### --- COMPUTE SILHOUETTE SCORES FOR PAIRED GENRES ---
def get_paired_genre_ratings(ratings, movies, genre_pairs):
    paired_ratings = pd.DataFrame()
    column_names = []
    
    for genre1, genre2 in genre_pairs:
        paired_movies = movies[movies['genres'].str.contains(genre1, na=False) & 
                               movies['genres'].str.contains(genre2, na=False)]
        avg_paired_votes_per_user = ratings[ratings['movieId'].isin(paired_movies['movieId'])] \
            .groupby('userId')['rating'].mean().round(2)
        paired_ratings = pd.concat([paired_ratings, avg_paired_votes_per_user], axis=1)
        column_names.append(f'avg_{genre1.lower()}_{genre2.lower()}_rating')
    
    paired_ratings.columns = column_names
    return paired_ratings

# Genre pairs to evaluate
genre_pairs = [('Romance', 'Drama'), ('Sci-Fi', 'Fantasy'), ('Action', 'Adventure')]

paired_genre_ratings = get_paired_genre_ratings(ratings, movies, genre_pairs)
paired_dataset = paired_genre_ratings.dropna()
X = paired_dataset.values

# Compute silhouette scores for Canberra-based clustering
predictions_canberra, _ = custom_kmeans_canberra(X, n_clusters=chosen_k)
silhouette_canberra = silhouette_score(X, predictions_canberra, metric='canberra')

# Compute silhouette scores for Euclidean-based clustering
predictions_euclidean = kmeans_original.fit_predict(X)
silhouette_euclidean = silhouette_score(X, predictions_euclidean, metric='euclidean')

# Compute silhouette scores for each genre pair
for i, (genre1, genre2) in enumerate(genre_pairs):
    genre_pair_data = paired_dataset.iloc[:, [i]].values
    score_canberra = silhouette_score(genre_pair_data, predictions_canberra, metric='canberra')
    score_euclidean = silhouette_score(genre_pair_data, predictions_euclidean, metric='euclidean')
    print(f"- {genre1} and {genre2} (Canberra): {score_canberra:.4f}")
    print(f"- {genre1} and {genre2} (Euclidean): {score_euclidean:.4f}")

### --- PLOT SIDE-BY-SIDE SCATTERPLOTS ---
def plot_side_by_side(genre_x, genre_y):
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    
    # Original K-Means Scatterplot
    ax = axes[0]
    for cluster in sorted(user_genre_ratings["Cluster_Original"].unique()):
        subset = user_genre_ratings[user_genre_ratings["Cluster_Original"] == cluster]
        ax.scatter(subset[genre_x], subset[genre_y], label=f'Cluster {cluster}', alpha=0.7)
    ax.set_xlabel(f'Average {genre_x} Rating')
    ax.set_ylabel(f'Average {genre_y} Rating')
    ax.set_title(f'Original K-Means (Euclidean) - {genre_x} vs {genre_y}')
    ax.legend()

    # Enhanced K-Means Scatterplot
    ax = axes[1]
    for cluster in sorted(user_genre_ratings["Cluster_Enhanced"].unique()):
        subset = user_genre_ratings[user_genre_ratings["Cluster_Enhanced"] == cluster]
        ax.scatter(subset[genre_x], subset[genre_y], label=f'Cluster {cluster}', alpha=0.7)
    ax.set_xlabel(f'Average {genre_x} Rating')
    ax.set_ylabel(f'Average {genre_y} Rating')
    ax.set_title(f'Enhanced K-Means (Canberra) - {genre_x} vs {genre_y}')
    ax.legend()

    plt.show()

for genre_x, genre_y in genre_pairs:
    plot_side_by_side(genre_x, genre_y)
