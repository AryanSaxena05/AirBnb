# Airbnb Host Segmentation & Superhost Prediction

This repository contains a complete, modular pipeline for Airbnb host segmentation and superhost prediction using machine learning. The codebase is organized for clarity, reproducibility, and ease of use.

## Project Structure

- `data/` — Raw and processed data files (add your CSVs here)
- `notebooks/` — Jupyter Notebooks for EDA, modeling, and reporting
- `src/` — Python modules for data processing, modeling, and utilities
- `outputs/` — Generated outputs (figures, cluster summaries, predictions)
- `requirements.txt` — Python dependencies
- `README.md` — Project overview and instructions

## Quick Start

1. Clone the repository and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Place your data files in the `data/` directory.
3. Run the notebooks in `notebooks/` for EDA, clustering, and prediction.
4. Use the scripts in `src/` for modular, reusable code.

## Main Steps

1. **Preprocessing & Visualization:**
   - Clean and explore the data, handle missing values, visualize distributions and correlations.
2. **Market Segmentation:**
   - Cluster hosts using KMeans based on relevant features.
   - Analyze and visualize cluster characteristics.
3. **Superhost Prediction:**
   - Train and evaluate machine learning models to predict superhost status.
   - Analyze feature importances globally and per cluster.
4. **Reporting:**
   - Summarize findings and generate visualizations for presentation.

## Dependencies
See `requirements.txt` for a full list.

## Authors
Team 40, MGMT 687
