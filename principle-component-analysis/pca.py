import kagglehub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

TRAINING_DATA_ALLOCATED = 0.8
K_TOP_PRINCIPLE_COMPONENTS = 2 #1 or 2

def get_dataset() -> tuple[np.ndarray, np.ndarray]:
    path = kagglehub.dataset_download("uciml/iris")
    df = pd.read_csv(f"{path}/iris.csv")
    
    # Drop Id
    if 'Id' in df.columns:
        df = df.drop(columns=['Id'])
    
    # Change labels to numbers
    df['Species'] = df['Species'].map({
        "Iris-setosa": 0,
        "Iris-versicolor": 1,
        "Iris-virginica": 2
    })
    
    # Make into matrix and shuffle
    matrix = df.to_numpy()
    np.random.seed(42) #take off once done for variability
    np.random.shuffle(matrix)
    
    # Split into features and labels
    features = matrix[:, :-1]
    labels = matrix[:, -1].astype(int)
    return features, labels

def split_data(features, labels, train_ratio) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_samples = features.shape[0]
    split_index = int(n_samples * train_ratio)

    X_train = features[:split_index]
    y_train = labels[:split_index]

    X_test = features[split_index:]
    y_test = labels[split_index:]

    return X_train, X_test, y_train, y_test

def normalize_features(X_train: np.ndarray, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Normalize features: mean-correct each column using only training data
    col_means = np.mean(X_train, axis=0)
    X_train_norm = X_train - col_means
    X_test_norm = X_test - col_means
    return X_train_norm, X_test_norm

def get_covariance(training_features) -> np.ndarray:
    # Compute covariance matrix
    n_samples = training_features.shape[0]
    return (training_features.T @ training_features) / (n_samples - 1)

def get_eigens(covariance_matrix) -> tuple[np.ndarray, np.ndarray]:
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

    # Sort by descending eigenvalue magnitude
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    return eigenvectors, eigenvalues

def multiply_by_top_eigenvectors(X_train, X_test, eigenvectors, k) -> tuple[np.ndarray, np.ndarray]:
    X_train_reduced_dim = X_train @ eigenvectors[:, :k]
    X_test_reduced_dim = X_test @ eigenvectors[:, :k]
    return X_train_reduced_dim, X_test_reduced_dim

def graph_data(X_reduced, y, title):
    plt.figure(figsize=(8, 6))
    for label in np.unique(y):
        mask = y == label
        plt.scatter(
            X_reduced[mask, 0], X_reduced[mask, 1],
            label=f"Class {label}", alpha=0.7
        )

    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    """
    ***Principle Component Analysis Steps***
    0. Get Data and split into training/testing and features/labels
    1. Normalize Data
    2. Calculate Covariance Matrix 
    3. Find Eigenvectors and Eigenvalues
    4. Select Principle Components
    5. Project Data in Low Dimension

    (We will use the Iris Dataset)
    """
    # Step 0
    features, labels = get_dataset()
    X_train, X_test, y_train, y_test = split_data(features, labels, TRAINING_DATA_ALLOCATED)
    print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}, y_test shape: {y_test.shape}")

    # Step 1
    X_train, X_test = normalize_features(X_train, X_test)

    # Step 2
    covariance_matrix = get_covariance(X_train)

    # Step 3
    eigenvectors, eigenvalues = get_eigens(covariance_matrix)

    # Step 4 & 5
    X_train_reduced_dim, X_test_reduced_dim = multiply_by_top_eigenvectors(
        X_train, X_test, eigenvectors, K_TOP_PRINCIPLE_COMPONENTS
    )

    print(f"Reduced X_train shape: {X_train_reduced_dim.shape}")
    print(f"Reduced X_test shape: {X_test_reduced_dim.shape}")

    # Visualize
    graph_data(X_train_reduced_dim, y_train, title="PCA of Iris Dataset (Train Set)")
    graph_data(X_test_reduced_dim, y_test, title="PCA of Iris Dataset (Test Set)")

if __name__ == "__main__":
    main()
