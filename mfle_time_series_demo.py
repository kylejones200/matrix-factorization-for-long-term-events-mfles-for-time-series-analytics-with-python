import logging

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import TruncatedSVD


def main():
    np.random.seed(42)

    # Simulate multivariate time series data

    # Configure logging
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    n_series = 10
    n_timesteps = 100
    time = np.linspace(0, 10, n_timesteps)

    data = np.array(
        [
            np.sin(time + phase) + np.random.normal(0, 0.3, n_timesteps)
            for phase in np.linspace(0, 2 * np.pi, n_series)
        ]
    )

    # Apply matrix factorization (SVD)
    svd = TruncatedSVD(n_components=3)
    latent_features = svd.fit_transform(data)

    # Reconstruct time series from latent features
    reconstructed = svd.inverse_transform(latent_features)

    # Plot original vs reconstructed for a few time series
    plt.figure(figsize=(12, 8))
    for i in range(3):
        plt.subplot(3, 1, i + 1)
        plt.plot(data[i], label="Original")
        plt.plot(reconstructed[i], label="Reconstructed", linestyle="--")
        plt.title(f"Time Series {i + 1}")
        plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
