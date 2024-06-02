# Movie Recommendation System

This project is a **Movie Recommendation System** using collaborative filtering. The system recommends movies to users based on their previous ratings and the ratings of similar users. This project demonstrates data preprocessing, model training using matrix factorization, and evaluation.

## Steps

### 1. Data Collection
- The dataset used for this project is the **MovieLens** dataset.
- This dataset contains movie ratings provided by users.

### 2. Data Preprocessing
- Data is preprocessed by creating a user-item matrix from the ratings.
- Missing values are filled with zeros.
- Matrix mean normalization is performed to prepare the data for Singular Value Decomposition (SVD).

### 3. Model Training
- **Singular Value Decomposition (SVD)** is used for collaborative filtering.
- The matrix is decomposed into three matrices: `U`, `sigma`, and `Vt`.

### 4. Prediction
- Predicted ratings are calculated by multiplying the decomposed matrices.
- A function is provided to recommend movies to a user based on these predictions.

### 5. Example Usage
- The function `recommend_movies` is used to get movie recommendations for a specific user.


**NOTE: To Upload File from Google Drive**:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   from google.colab import files
   uploaded = files.upload()
   File_Name= pd.read_csv('/content/File_Name.csv')
   
