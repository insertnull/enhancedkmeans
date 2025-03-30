import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import calinski_harabasz_score, pairwise_distances

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

# Standardize data
scaler = StandardScaler()
user_genre_ratings_scaled = scaler.fit_transform(user_genre_ratings)

# Remove outliers using LOF
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05, metric="canberra")
outliers_lof = lof.fit_predict(user_genre_ratings_scaled)
user_genre_ratings_filtered = user_genre_ratings[outliers_lof == 1]
user_genre_ratings_filtered_scaled = scaler.fit_transform(user_genre_ratings_filtered)

# Custom K-Means using Canberra distance
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

def recommend_movies(user_id):
    if user_id not in user_genre_ratings_filtered.index:
        messagebox.showerror("Error", "User ID not found!")
        return [], []

    user_cluster = user_genre_ratings_filtered.loc[user_id, 'Cluster_Enhanced']
    similar_users = user_genre_ratings_filtered[user_genre_ratings_filtered['Cluster_Enhanced'] == user_cluster].index
    
    # Get movies rated by the user
    user_movies = df[df['userId'] == user_id]

    # Get the top-rated movies by the user (rating >= 4.0), limited to top 10
    highly_rated_movies = user_movies[user_movies['rating'] >= 4.0].sort_values(by="rating", ascending=False)
    highly_rated_movies_list = highly_rated_movies['title'].head(10).tolist()

    # Get cluster movies excluding the ones user already watched
    cluster_movies = df[df['userId'].isin(similar_users) & ~df['movieId'].isin(user_movies['movieId'])]

    # Recommend top movies based on average rating in the cluster
    recommended_movies = cluster_movies.groupby('movieId')['rating'].mean().sort_values(ascending=False).head(10).index
    recommended_movies_list = movies[movies['movieId'].isin(recommended_movies)]['title'].tolist()

    return recommended_movies_list, highly_rated_movies_list

# Tkinter UI
def on_recommend():
    user_id = int(user_dropdown.get())
    recommended, highly_rated = recommend_movies(user_id)

    result = "Recommended Movies:\n" + ("\n".join(recommended) if recommended else "No recommendations found.") + "\n\n"
    result += "Highly Rated by User:\n" + ("\n".join(highly_rated) if highly_rated else "No highly rated movies found.")
    
    result_text.set(result)

root = tk.Tk()
root.title("Movie Recommendation System")

frame = ttk.Frame(root, padding=10)
frame.grid(row=0, column=0)

label = ttk.Label(frame, text="Select User:")
label.grid(row=0, column=0)

user_dropdown = ttk.Combobox(frame, values=list(user_genre_ratings_filtered.index))
user_dropdown.grid(row=0, column=1)
user_dropdown.current(0)

recommend_button = ttk.Button(frame, text="Get Recommendations", command=on_recommend)
recommend_button.grid(row=1, columnspan=2)

result_text = tk.StringVar()
result_label = ttk.Label(frame, textvariable=result_text, wraplength=400, justify="left")
result_label.grid(row=2, columnspan=2)

root.mainloop()
