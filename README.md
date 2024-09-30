# Dynamic Price Prediction Engine

This project predicts future prices of iPhones and Samsung devices sold on eBay using a machine learning model based on LSTM (Long Short-Term Memory) neural networks. The model was trained on real historical price data from `iphone_ebay.csv` and `samsung_ebay.csv`.

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running the Project](#running-the-project)


## Project Overview

The **Dynamic Price Prediction Engine** is designed to track historical price trends and predict future prices of popular electronics. Using real e-commerce data from eBay, this project cleans, preprocesses, and models time-series data to forecast prices using LSTM-based neural networks.

## Features

- **Data Preprocessing**: Cleans and processes price data, handling inconsistencies like price ranges and missing values.
- **Feature Engineering**: Extracts meaningful features from product names, such as model, storage capacity, and color, while incorporating time-based features like day of the week and month.
- **LSTM Model**: Builds an LSTM neural network to predict future prices using historical price data.
- **Visualization**: Generates clear plots comparing actual vs. predicted prices, enabling easy analysis of the model's performance.

## Technologies Used

- **Python 3.12**
- **Libraries**: Pandas, NumPy, Matplotlib, Seaborn, PyTorch, Scikit-learn
- **Machine Learning Framework**: PyTorch for building LSTM models

## Getting Started

### Prerequisites

Make sure you have the following installed:

- Python 3.12 or higher
- pip (Python package installer)

### Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/adankhalid1/dynamic-price-prediction-engine.git
   cd dynamic-price-prediction-engine

2. Create a Virtual Environment (Optional but Recommended)

   ```bash
   python -m venv env
   env\Scripts\activate #Windows
   source env/bin/activate #MacOS/Linux

3. Install the necessary Python libraries:
   ```bash
   pip install pandas numpy matplotlib seaborn torch scikit-learn

## Running the Project

1. Place Your Datasets

Download iphone_ebay.csv and samsung_ebay.csv and place them in the data/ directory.

2. Run the Python Script
   ```bash
   python price_prediction.py

