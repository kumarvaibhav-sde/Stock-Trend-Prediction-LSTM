# Stock Trend Prediction Using LSTM — Machine Learning & Flask Web Application

An end-to-end stock price prediction web application built using Python, Flask, TensorFlow, and Long Short-Term Memory (LSTM) neural networks.

The project uses historical stock market data to predict closing prices, visualize market trends, and evaluate prediction performance through multiple regression metrics.

## Project Overview

Stock Trend Prediction is a machine learning-based application designed to analyze historical stock market data and forecast future closing prices using an LSTM deep learning model.

The application retrieves historical stock data through the Yahoo Finance API, preprocesses the data, generates predictions using a pre-trained LSTM model, and displays results through an interactive Flask web interface.

## Key Features

- Historical stock data retrieval using Yahoo Finance (`yfinance`).
- Data preprocessing and Min-Max normalization.
- LSTM-based stock price prediction using 100-day historical sequences.
- Exponential Moving Average (EMA) analysis: 20, 50, 100, and 200 days.
- Actual vs. predicted stock price visualization.
- Model evaluation using MAE, MSE, RMSE, MAPE, R² Score, and Directional Accuracy.
- Descriptive statistical analysis of stock data.
- Interactive Flask web application.
- Downloadable historical stock dataset in CSV format.

## Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Deep Learning | TensorFlow, Keras, LSTM |
| Data Analysis | Pandas, NumPy |
| Data Visualization | Matplotlib |
| Machine Learning | Scikit-learn |
| Stock Data API | Yahoo Finance (yfinance) |
| Backend | Flask |
| Frontend | HTML, CSS|

## Methodology

The project follows these major steps:

1. **Data Acquisition:** Retrieve historical stock market data using Yahoo Finance.
2. **Data Preprocessing:** Extract closing prices and prepare the time-series dataset.
3. **Train-Test Splitting:** Divide the dataset chronologically into 70% training and 30% testing data.
4. **Feature Scaling:** Apply Min-Max normalization to scale stock prices between 0 and 1.
5. **Sequence Generation:** Create input sequences using the previous 100 trading observations.
6. **LSTM Prediction:** Load the trained LSTM model and generate predictions.
7. **Inverse Transformation:** Convert normalized predictions back into original stock-price values.
8. **Performance Evaluation:** Calculate regression and directional evaluation metrics.
9. **Visualization:** Generate EMA and actual-versus-predicted stock price charts.
10. **Result Presentation:** Display predictions, metrics, descriptive statistics, and downloadable data through Flask.

## Model Evaluation Metrics

The application evaluates model performance using:

- **MAE:** Mean Absolute Error
- **MSE:** Mean Squared Error
- **RMSE:** Root Mean Squared Error
- **MAPE:** Mean Absolute Percentage Error
- **R² Score:** Coefficient of Determination
- **Directional Accuracy:** Percentage of correctly predicted price movements

These metrics provide different perspectives on prediction error and trend direction. R² should not be interpreted as prediction accuracy.

## Project Structure

```text
Stock-Trend-Prediction-LSTM/
│
├── app.py
├── stock_dl_model.h5
├── requirements.txt
├── Procfile
├── README.md
├── .gitignore
│
├── templates/
│   └── index.html
│
└── static/
    ├── ema_20_50.png
    ├── ema_100_200.png
    └── stock_prediction.png
```

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/kumarvaibhav-sde/Stock-Trend-Prediction-LSTM.git
```

### 2. Navigate to the Project Directory

```bash
cd Stock-Trend-Prediction-LSTM
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Application

```bash
python app.py
```

### 6. Open in Browser

```text
http://127.0.0.1:5000/
```

## Deployment

The application can be deployed using cloud hosting platforms that support Python and Flask.

Suggested deployment options:

- GitHub – Source code management and version control.
- Render – Flask application hosting.

For deployment, configure Gunicorn as the production web server and provide the required Python dependencies.

## Limitations

- Stock prices are influenced by unpredictable market conditions.
- Historical trends do not guarantee future performance.
- Prediction quality depends on the training dataset and model configuration.
- The application uses a pre-trained LSTM model and does not retrain it for every submitted ticker.
- Free hosting services may have memory, execution-time, and sleep limitations.

## Future Enhancements

- Hyperparameter optimization for LSTM.
- Integration of additional technical indicators.
- Sentiment analysis using financial news.
- Comparison of LSTM, GRU, and hybrid models.
- Walk-forward validation and improved time-series evaluation.
- Cloud deployment and performance optimization.

## Disclaimer

This project is developed for educational and research purposes only. Its predictions are not financial advice or guaranteed investment recommendations. Always conduct independent research before making investment decisions.

## Author

**Kumar Vaibhav**

Machine Learning | Python | Data Analytics | Deep Learning

GitHub: https://github.com/kumarvaibhav-sde
Linkedin: https://linkedin.com/in/kumarvaibhav-sde

---

⭐ If you find this project useful, consider giving the repository a star!
