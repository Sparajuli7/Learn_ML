# Time Series Forecasting

## Overview
Time series forecasting is a specialized branch of machine learning that deals with predicting future values based on historical data points collected over time. It's essential for business planning, financial modeling, weather prediction, and many other domains.

## Key Concepts

### Time Series Components
- **Trend**: Long-term increase or decrease in the data
- **Seasonality**: Repeating patterns at regular intervals
- **Cyclical**: Long-term oscillations without fixed period
- **Noise**: Random fluctuations in the data

### Stationarity
A time series is stationary if its statistical properties (mean, variance, autocorrelation) remain constant over time.

**Tests for Stationarity:**
- Augmented Dickey-Fuller (ADF) test
- Kwiatkowski-Phillips-Schmidt-Shin (KPSS) test

## Traditional Methods

### 1. Moving Averages
```python
# Simple Moving Average
def simple_moving_average(data, window):
    return data.rolling(window=window).mean()

# Exponential Moving Average
def exponential_moving_average(data, alpha):
    return data.ewm(alpha=alpha).mean()
```

### 2. ARIMA Models
**ARIMA (AutoRegressive Integrated Moving Average)** combines:
- **AR (p)**: Autoregression of order p
- **I (d)**: Integration of order d
- **MA (q)**: Moving average of order q

```python
from statsmodels.tsa.arima.model import ARIMA

# Fit ARIMA model
model = ARIMA(data, order=(p, d, q))
fitted_model = model.fit()
forecast = fitted_model.forecast(steps=12)
```

### 3. Seasonal Decomposition
```python
from statsmodels.tsa.seasonal import seasonal_decompose

# Decompose time series
decomposition = seasonal_decompose(data, model='additive')
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid
```

## Modern Deep Learning Approaches

### 1. Recurrent Neural Networks (RNN)
```python
import torch
import torch.nn as nn

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_size)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out
```

### 2. Long Short-Term Memory (LSTM)
```python
class LSTMForecaster(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMForecaster, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out
```

### 3. Temporal Fusion Transformers (TFT)
Advanced architecture that handles multiple time series with different frequencies and external variables.

### 4. Prophet (Facebook)
```python
from fbprophet import Prophet

# Create and fit model
model = Prophet()
model.fit(df)
future = model.make_future_dataframe(periods=365)
forecast = model.predict(future)
```

## Feature Engineering for Time Series

### 1. Lag Features
```python
# Create lag features
df['lag_1'] = df['value'].shift(1)
df['lag_7'] = df['value'].shift(7)
df['lag_30'] = df['value'].shift(30)
```

### 2. Rolling Statistics
```python
# Rolling mean and std
df['rolling_mean_7'] = df['value'].rolling(window=7).mean()
df['rolling_std_7'] = df['value'].rolling(window=7).std()
df['rolling_max_7'] = df['value'].rolling(window=7).max()
```

### 3. Time-Based Features
```python
# Extract time components
df['hour'] = df.index.hour
df['day_of_week'] = df.index.dayofweek
df['month'] = df.index.month
df['quarter'] = df.index.quarter
```

## Evaluation Metrics

### 1. Mean Absolute Error (MAE)
```python
def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))
```

### 2. Mean Squared Error (MSE)
```python
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)
```

### 3. Root Mean Squared Error (RMSE)
```python
def rmse(y_true, y_pred):
    return np.sqrt(mse(y_true, y_pred))
```

### 4. Mean Absolute Percentage Error (MAPE)
```python
def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
```

## Practical Implementation

### Data Preprocessing Pipeline
```python
def preprocess_timeseries(data, target_col, sequence_length=60):
    # Normalize data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    
    # Create sequences
    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i])
        y.append(scaled_data[i, data.columns.get_loc(target_col)])
    
    return np.array(X), np.array(y), scaler
```

### Training Loop
```python
def train_forecasting_model(model, train_loader, val_loader, epochs=100):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch}: Train Loss: {train_loss/len(train_loader):.4f}, '
                  f'Val Loss: {val_loss/len(val_loader):.4f}')
```

## Advanced Techniques

### 1. Ensemble Methods
```python
def ensemble_forecast(models, X_test):
    predictions = []
    for model in models:
        pred = model.predict(X_test)
        predictions.append(pred)
    
    # Average predictions
    ensemble_pred = np.mean(predictions, axis=0)
    return ensemble_pred
```

### 2. Multi-Step Forecasting
```python
def multi_step_forecast(model, initial_sequence, steps):
    predictions = []
    current_sequence = initial_sequence.copy()
    
    for _ in range(steps):
        # Predict next value
        next_pred = model.predict(current_sequence.reshape(1, -1))
        predictions.append(next_pred[0])
        
        # Update sequence
        current_sequence = np.roll(current_sequence, -1)
        current_sequence[-1] = next_pred[0]
    
    return predictions
```

### 3. Uncertainty Quantification
```python
def monte_carlo_forecast(model, X_test, n_samples=1000):
    predictions = []
    for _ in range(n_samples):
        # Add noise to input
        noisy_X = X_test + np.random.normal(0, 0.01, X_test.shape)
        pred = model.predict(noisy_X)
        predictions.append(pred)
    
    mean_pred = np.mean(predictions, axis=0)
    std_pred = np.std(predictions, axis=0)
    
    return mean_pred, std_pred
```

## Applications

### 1. Financial Forecasting
- Stock price prediction
- Currency exchange rates
- Commodity prices

### 2. Demand Forecasting
- Retail sales
- Energy consumption
- Transportation demand

### 3. Weather Prediction
- Temperature forecasting
- Precipitation prediction
- Wind speed estimation

### 4. Healthcare
- Patient monitoring
- Disease outbreak prediction
- Medical resource planning

## Best Practices

1. **Data Quality**: Ensure clean, consistent data with proper handling of missing values
2. **Feature Engineering**: Create relevant lag features and rolling statistics
3. **Cross-Validation**: Use time series cross-validation (TimeSeriesSplit)
4. **Model Selection**: Compare multiple models and ensemble approaches
5. **Regular Retraining**: Update models periodically with new data
6. **Domain Knowledge**: Incorporate business insights and seasonal patterns

## Tools and Libraries

- **Statsmodels**: Traditional time series models
- **Prophet**: Facebook's forecasting tool
- **PyTorch/TensorFlow**: Deep learning models
- **scikit-learn**: Feature engineering and preprocessing
- **pandas**: Data manipulation
- **numpy**: Numerical computations

## Next Steps

1. **Advanced Models**: Explore Transformer-based architectures for time series
2. **Multivariate Forecasting**: Handle multiple related time series
3. **Real-time Forecasting**: Implement streaming prediction systems
4. **Causal Inference**: Understand causal relationships in time series data
5. **Anomaly Detection**: Identify unusual patterns in time series data

---

*Time series forecasting combines statistical rigor with modern machine learning techniques to predict future values based on historical patterns. Mastery of both traditional methods and deep learning approaches is essential for building robust forecasting systems.* 