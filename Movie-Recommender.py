import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from scipy.sparse.linalg import svds

from google.colab import drive
drive.mount('/content/drive')

from google.colab import files
uploaded = files.upload()

ratings = pd.read_csv('/content/ratings.csv')

# Data preprocessing
user_item_matrix = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)
matrix = user_item_matrix.values
user_ratings_mean = np.mean(matrix, axis=1)
matrix_demeaned = matrix - user_ratings_mean.reshape(-1, 1)

# Singular Value Decomposition
U, sigma, Vt = svds(matrix_demeaned, k=50)
sigma = np.diag(sigma)

# Predict ratings
predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)
predicted_df = pd.DataFrame(predicted_ratings, columns=user_item_matrix.columns)

# Function to recommend movies
def recommend_movies(user_id, num_recommendations=5):
    user_idx = user_id - 1
    sorted_user_ratings = predicted_df.iloc[user_idx].sort_values(ascending=False)
    recommendations = sorted_user_ratings.head(num_recommendations).index
    return recommendations

# Example usage
print(recommend_movies(user_id=1))
