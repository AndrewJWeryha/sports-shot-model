# Sports Shot Model (Logistic Regression)

A simple Python model to predict basketball shot make probabilities using tracking data (distance, defender proximity, and shot clock).

## Overview

This script cleans incoming tracking features, trains a standard logistic regression model using `scikit-learn`, and outputs shot make probabilities both to the console and to a CSV file.

- **Features used:**
  - `distance`: Distance from the basket (ft)
  - `closestdefdist`: Distance to nearest defender (ft)
  - `shotclock`: Seconds left on shot clock (s)
- **Output:** Saves predictions sorted by probability to `submission.csv`.

## Project Structure

```text
├── data/
│   ├── training.csv
│   └── testing.csv
├── LogisticRegression.py
└── README.md
