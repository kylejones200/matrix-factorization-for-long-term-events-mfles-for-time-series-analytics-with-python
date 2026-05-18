# Matrix Factorization for Long-Term Events (MFLE) for Time Series Analytics

This project demonstrates matrix factorization using Truncated SVD for time series analysis and reconstruction.

## Business context

Matrix Factorization for Long-term Events (MFLEs) is an advanced time series forecasting technique useful for large, multivariate datasets. As the name implies, it uses matrix factorization to extract latent features that can represent the underlying patterns in the data --- which is way more sophisticated than moving average.

Latent variable analysis has always intrigued me because it feels mystical. We always talk about finding hidden patterns in data but latent variables are a whole other level. MFLE makes since if you have a large number of time series that (you think) share underlying patterns. Highly dimensional data can be computational expensive and MFLE helps reduce dimensionality similar to how we use PCA for regression problems.

It works by decomposing the data matrix into latent components that represent the key patterns and relationships. This helps it captures changes over time by modeling time as a factor. And then it looks for common trends or seasonality across multiple time series.

## Article

Medium article: [Matrix Factorization for Long-Term Events](https://medium.com/@kylejones_47003/matrix-factorization-for-long-term-events-mfles-for-time-series-analytics-with-python-71aba4800c91)

## Project Structure

```
.
├── README.md           # This file
├── main.py            # Main entry point
├── config.yaml        # Configuration file
├── requirements.txt   # Python dependencies
├── src/               # Core functions
│   ├── core.py        # Matrix factorization functions
│   └── plotting.py    # Tufte-style plotting utilities
├── tests/             # Unit tests
├── data/              # Data files (if needed)
└── images/            # Generated plots and figures
```

## Configuration

Edit `config.yaml` to customize:
- Data generation parameters (n_series, n_timesteps, noise level)
- SVD parameters (n_components)
- Output settings

## Caveats

- By default, the script generates synthetic multivariate time series data.
- Truncated SVD reduces dimensionality while preserving variance.
- The number of components determines the compression ratio and reconstruction quality.

## Disclaimer

Educational/demo code only. Not financial, safety, or engineering advice. Use at your own risk. Verify results independently before any production or operational use.

## License

MIT — see [LICENSE](LICENSE).