import cv2
import glob
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time
import numpy as np
import os

def extract_image_features(source_path: str, label: str) -> None:
    """
    Extracts fundamental statistical features and quadrant means from raw images 
    to reduce dimensionality for the classification model. 
    Appends the engineered features directly to a local CSV dataset to build the training set.
    """
    features_list = []
    
    for file_path in glob.glob(source_path):
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        
        # Standardize image dimensions to ensure consistent feature extraction across the dataset
        resized_img = cv2.resize(img, (64, 64))

        h, w = resized_img.shape
        
        # Subdivide the image into quadrants to capture localized intensity patterns and shapes
        q1_mean = resized_img[0:h//2, 0:w//2].mean()
        q2_mean = resized_img[0:h//2, w//2:w].mean()
        q3_mean = resized_img[h//2:h, 0:w//2].mean()
        q4_mean = resized_img[h//2:h, w//2:w].mean() 

        # Calculate global statistical properties of the image
        img_median = np.median(resized_img)
        img_mode = calculate_numpy_mode(resized_img)
        val_max = resized_img.max()
        val_min = resized_img.min()
        std_dev = resized_img.std()

        features_list.append([
            img_median, img_mode, val_max, val_min, std_dev, 
            q1_mean, q2_mean, q3_mean, q4_mean, label
        ])

    df = pd.DataFrame(features_list)
    df.to_csv("dataset.csv", mode='a', index=False, header=False)

def calculate_numpy_mode(image_array: np.ndarray) -> int:
    """
    Determines the most frequent pixel intensity value.
    Implemented via numpy unique counts since standard numpy lacks a native mode function.
    """
    values, counts = np.unique(image_array, return_counts=True)
    max_index = np.argmax(counts)
    return values[max_index]

def find_optimal_k(X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray, cov_matrix: np.ndarray) -> int:
    """
    Performs a brute-force search to find the optimal number of neighbors (K) 
    that maximizes accuracy on the validation set, mitigating underfitting or overfitting.
    """
    best_k = 1
    max_acc = 0
    print("Searching for the optimal K 1-20")
    
    for k in range(1, 21):
        knn = KNeighborsClassifier(
            n_neighbors=k, 
            metric='mahalanobis', 
            metric_params={'V': cov_matrix}
        )
        knn.fit(X_train, y_train)
        
        predictions = knn.predict(X_test)
        acc = accuracy_score(y_test, predictions)
        
        if acc > max_acc:
            max_acc = acc
            best_k = k
            
    return best_k

def train_knn_model(dataset_path: str) -> KNeighborsClassifier:
    """
    Loads the engineered feature dataset, splits it into training and validation sets,
    and trains a K-Nearest Neighbors classifier using the Mahalanobis distance metric
    to account for correlations between the extracted statistical features.
    """
    data = pd.read_csv(dataset_path, header=None)
    
    # Separate input features (X) from the target classification labels (y)
    X = data.iloc[:, :-1].values   
    y = data.iloc[:, -1].values    
    
    print(f"Dataset loaded: {X.shape} features, {len(set(y))} classes")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=8)

    # Calculate covariance matrix required for Mahalanobis distance
    cov_matrix = np.cov(X_train, rowvar=False)
    
    # Add a small constant to the diagonal (Ridge regularization) to prevent singular matrix errors 
    # during distance calculation
    cov_matrix += np.eye(cov_matrix.shape[0]) * 0.1

    optimal_k = find_optimal_k(X_train, X_test, y_train, y_test, cov_matrix)
    print(f"Optimal K value found: {optimal_k}")

    knn = KNeighborsClassifier(
        n_neighbors=optimal_k,
        metric='mahalanobis',
        metric_params={'V': cov_matrix}
    )
    knn.fit(X_train, y_train)

    predictions = knn.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    print(f"KNN Model Accuracy (Mahalanobis): {acc*100:.2f}%")
    
    return knn

def main(): 
    """
    Orchestrates the data pipeline: cleans up legacy data, processes raw image 
    directories into tabular features, and trains the final classification model.
    """
    start_time = time.perf_counter()

    # Ensure a fresh dataset is generated for every execution to avoid appending to old runs
    if os.path.exists("dataset.csv"):
        os.remove("dataset.csv")
        print("Previous 'dataset.csv' removed.")

    # Extract features for each respective class directory
    extract_image_features('A/*.JPG', 'A')
    extract_image_features('B/*.JPG', 'B')
    extract_image_features('C/*.JPG', 'C')
    extract_image_features('D/*.JPG', 'D')
    
    train_knn_model("dataset.csv")
    
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    print(f"Execution took {elapsed_time:.2f} seconds")

if __name__ == '__main__':
    main()