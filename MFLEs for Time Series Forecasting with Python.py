
def main() -> None:
    # --- notebook cell ---
    """Generated from Jupyter notebook: MFLEs for Time Series Forecasting with Python

    Magics and shell lines are commented out. Run with a normal Python interpreter."""


    # --- code cell ---

    Requirements: 
    scikit-learn==1.3.2
    numpy==1.24.4
    pandas==2.0.3
    matplotlib==3.6.0


    # --- code cell ---

    import numpy as np
    import pandas as pd
    from sklearn.decomposition import TruncatedSVD
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt

    # Generate synthetic data
    np.random.seed(42)
    n_series = 100  # Number of time series
    n_timepoints = 50  # Number of time points

    # Simulated data matrix (rows: time series, columns: time points)
    data_matrix = np.random.rand(n_series, n_timepoints)
    df = pd.DataFrame(data_matrix)
    print(df.head())

    # Decompose the data matrix
    svd = TruncatedSVD(n_components=10)  # Reduce to 10 latent features
    latent_features = svd.fit_transform(data_matrix)

    # Reconstruct the time series
    reconstructed_matrix = svd.inverse_transform(latent_features)

    # Prepare training and test datasets
    X = latent_features[:, :-1]
    y = latent_features[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)


    # --- code cell ---

    import matplotlib.pyplot as plt
    import seaborn as sns

    # Set up the plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Time Series Analysis and Prediction', fontsize=16)

    # 1. Original vs. Reconstructed Data (for the first time series)
    axes[0, 0].plot(data_matrix[:1].T, 'b-', alpha=0.5, label='Original')
    axes[0, 0].plot(reconstructed_matrix[:1].T, color="Red", label='Reconstructed')
    axes[0, 0].set_title('Original vs. Reconstructed Data')
    axes[0, 0].set_xlabel('Time Points')
    axes[0, 0].set_ylabel('Value')
    axes[0, 0].legend()

    # 2. Explained Variance Ratio
    explained_variance_ratio = svd.explained_variance_ratio_
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
    axes[0, 1].plot(range(1, len(explained_variance_ratio) + 1), cumulative_variance_ratio, 'bo-')
    axes[0, 1].set_title('Cumulative Explained Variance Ratio')
    axes[0, 1].set_xlabel('Number of Components')
    axes[0, 1].set_ylabel('Cumulative Explained Variance Ratio')
    axes[0, 1].set_ylim([0, 1])

    # 3. Actual vs. Predicted Values
    axes[1, 0].scatter(y_test, y_pred)
    axes[1, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[1, 0].set_title('Actual vs. Predicted Values')
    axes[1, 0].set_xlabel('Actual Values')
    axes[1, 0].set_ylabel('Predicted Values')

    # 4. Residual Plot
    residuals = y_test - y_pred
    axes[1, 1].scatter(y_pred, residuals)
    axes[1, 1].axhline(y=0, color='r', linestyle='--')
    axes[1, 1].set_title('Residual Plot')
    axes[1, 1].set_xlabel('Predicted Values')
    axes[1, 1].set_ylabel('Residuals')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
