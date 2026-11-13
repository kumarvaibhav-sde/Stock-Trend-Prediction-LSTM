import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras.models import load_model
from flask import Flask, render_template, request, send_file
import datetime as dt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import os
plt.style.use("fivethirtyeight")
from curl_cffi import requests as curl_requests
#Accuracy
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    r2_score
)



app = Flask(__name__)

# Load the model (make sure your model is in the correct path)
model = load_model('stock_dl_model.h5')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        stock = request.form.get('stock')
        if not stock:
            stock = 'AAPL'  # Default stock if none is entered
        
        # Define the start and end dates for stock data
        start = dt.datetime(2000, 1, 1)
        end = dt.datetime(2026, 11, 1)
        
        # Download stock data
        session = curl_requests.Session(impersonate="firefox")
        
        df = yf.download(
            stock,
            start=start,
            end=end,
            session=session,
            progress=False,
            auto_adjust=True
        )
        
        if df is None or df.empty:
            return "Unable to fetch stock data. Please try again later.", 400
        
        if len(df) < 334:
            return "Insufficient historical stock data.", 400

        ####
        
        # Descriptive Data
        data_desc = df.describe()
        
        # Exponential Moving Averages
        ema20 = df.Close.ewm(span=20, adjust=False).mean()
        ema50 = df.Close.ewm(span=50, adjust=False).mean()
        ema100 = df.Close.ewm(span=100, adjust=False).mean()
        ema200 = df.Close.ewm(span=200, adjust=False).mean()
        
        # Data splitting
        data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
        data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])
        
        # Scaling data
        """scaler = MinMaxScaler(feature_range=(0, 1))
        data_training_array = scaler.fit_transform(data_training)"""

        # Data splitting: chronological 70:30
        close_data = df['Close']

        # Handle yfinance returning a DataFrame for Close
        if isinstance(close_data, pd.DataFrame):
            close_data = close_data.iloc[:, 0]

        data_training = pd.DataFrame(
            close_data.iloc[:int(len(close_data) * 0.70)]
        )

        data_testing = pd.DataFrame(
            close_data.iloc[int(len(close_data) * 0.70):]
        )

        # Fit scaler ONLY on training data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(data_training)

        # Add the previous 100 training days to test data
        past_100_days = data_training.tail(100)

        final_df = pd.concat(
            [past_100_days, data_testing],
            ignore_index=True
        )

        # Transform test data using the already-fitted scaler
        #input_data = scaler.transform(final_df)
        input_data = scaler.transform(final_df)
        # Prepare X_test and y_test
        x_test = []
        y_test = []

        for i in range(100, input_data.shape[0]):
            x_test.append(input_data[i - 100:i])
            y_test.append(input_data[i, 0])

        x_test = np.array(x_test)
        y_test = np.array(y_test).reshape(-1, 1)

        # Generate predictions
        y_predicted = model.predict(x_test, verbose=0)

        # Convert predictions and actual values back to stock prices
        y_predicted = scaler.inverse_transform(y_predicted)
        y_test = scaler.inverse_transform(y_test)

        # Flatten arrays
        actual = y_test.flatten()
        predicted = y_predicted.flatten()

        # Calculate evaluation metrics
        mae = mean_absolute_error(actual, predicted)

        mse = mean_squared_error(actual, predicted)

        rmse = np.sqrt(mse)

        mape = mean_absolute_percentage_error(
            actual, predicted
        ) * 100

        r2 = r2_score(actual, predicted)

        # Calculate Directional Accuracy
        previous_actual = np.concatenate((
            data_training.iloc[-1:].values.flatten(),
            data_testing.iloc[:-1].values.flatten()
        ))

        actual_direction = np.sign(actual - previous_actual)
        predicted_direction = np.sign(predicted - previous_actual)

        directional_accuracy = np.mean(
            actual_direction == predicted_direction
        ) * 100

        # Store metrics for the frontend
        metrics = {
            "MAE": round(float(mae), 4),
            "MSE": round(float(mse), 4),
            "RMSE": round(float(rmse), 4),
            "MAPE": round(float(mape), 4),
            "R2": round(float(r2), 4),
            "Directional Accuracy": round(float(directional_accuracy), 2)
        }

        print("Model Evaluation Results:")
        print(metrics)

        
        # Prepare data for prediction
        past_100_days = data_training.tail(100)
        final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
        input_data = scaler.fit_transform(final_df)
        
        x_test, y_test = [], []
        for i in range(100, input_data.shape[0]):
            x_test.append(input_data[i - 100:i])
            y_test.append(input_data[i, 0])
        x_test, y_test = np.array(x_test), np.array(y_test)

        # Make predictions
        y_predicted = model.predict(x_test)
        
        # Inverse scaling for predictions
        scaler = scaler.scale_
        scale_factor = 1 / scaler[0]
        y_predicted = y_predicted * scale_factor
        y_test = y_test * scale_factor 

        
        # Plot 1: Closing Price vs Time Chart with 20 & 50 Days EMA
        fig1, ax1 = plt.subplots(figsize=(12, 6))
        ax1.plot(df.Close, 'y', label='Closing Price')
        ax1.plot(ema20, 'g', label='EMA 20')
        ax1.plot(ema50, 'r', label='EMA 50')
        ax1.set_title("Closing Price vs Time (20 & 50 Days EMA)")
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Price")
        ax1.legend()
        ema_chart_path = "static/ema_20_50.png"
        fig1.savefig(ema_chart_path)
        plt.close(fig1)
        
        # Plot 2: Closing Price vs Time Chart with 100 & 200 Days EMA
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        ax2.plot(df.Close, 'y', label='Closing Price')
        ax2.plot(ema100, 'g', label='EMA 100')
        ax2.plot(ema200, 'r', label='EMA 200')
        ax2.set_title("Closing Price vs Time (100 & 200 Days EMA)")
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Price")
        ax2.legend()
        ema_chart_path_100_200 = "static/ema_100_200.png"
        fig2.savefig(ema_chart_path_100_200)
        plt.close(fig2)
        
        # Plot 3: Prediction vs Original Trend
        fig3, ax3 = plt.subplots(figsize=(12, 6))
        ax3.plot(y_test, 'g', label="Original Price", linewidth = 1)
        ax3.plot(y_predicted, 'r', label="Predicted Price", linewidth = 1)
        ax3.set_title("Prediction vs Original Trend")
        ax3.set_xlabel("Time")
        ax3.set_ylabel("Price")
        ax3.legend()
        prediction_chart_path = "static/stock_prediction.png"
        fig3.savefig(prediction_chart_path)
        plt.close(fig3)



        
        # Save dataset as CSV
        csv_file_path = f"static/{stock}_dataset.csv"
        df.to_csv(csv_file_path)

        # Return the rendered template with charts and dataset
        '''return render_template('index.html', 
                               plot_path_ema_20_50=ema_chart_path, 
                               plot_path_ema_100_200=ema_chart_path_100_200, 
                               plot_path_prediction=prediction_chart_path, 
                               data_desc=data_desc.to_html(classes='table table-bordered'),
                               dataset_link=csv_file_path)'''
                                
        return render_template('index.html',
                                plot_path_ema_20_50=ema_chart_path,
                                plot_path_ema_100_200=ema_chart_path_100_200,
                                plot_path_prediction=prediction_chart_path,
                                data_desc=data_desc.to_html(classes='table table-bordered'),
                                dataset_link=csv_file_path,
                                metrics=metrics
)
    
    return render_template('index.html')

@app.route('/download/<filename>')
def download_file(filename):
    return send_file(f"static/{filename}", as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
