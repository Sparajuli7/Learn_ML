# AI in Energy: Smart Grids & Renewable Optimization

*"Powering the future with intelligent energy systems"*

---

## 📚 Table of Contents

1. [Introduction](#introduction)
2. [Mathematical Foundations](#mathematical-foundations)
3. [Implementation](#implementation)
4. [Applications](#applications)
5. [Exercises and Projects](#exercises-and-projects)
6. [Further Reading](#further-reading)

---

## 🎯 Introduction

The energy sector is undergoing a massive transformation in 2025: decarbonization, renewable integration, smart grids, and energy storage. AI is the key enabler for this transition, optimizing everything from power generation to consumption.

### Historical Context

Traditional energy systems were centralized and predictable. The energy transition introduces:
- **Distributed Generation**: Solar, wind, and storage at consumer level
- **Variable Renewables**: Intermittent power sources requiring intelligent management
- **Smart Grids**: Two-way communication and real-time optimization
- **Energy Storage**: Batteries, pumped hydro, and emerging technologies

### 2025 Energy Landscape

**Global Challenges:**
- Net-zero emissions by 2050
- 50% renewable energy by 2030
- Grid stability with variable renewables
- Energy storage optimization
- Demand response integration

**AI Solutions:**
- Renewable energy forecasting
- Smart grid optimization
- Energy storage management
- Demand-side management
- Grid stability prediction

---

## 🧮 Mathematical Foundations

### 1. Renewable Energy Forecasting

**Wind Power Prediction Model:**

```
P(t) = ½ × ρ × A × Cp(λ,β) × v³(t)
```

Where:
- P(t) = Power output at time t
- ρ = Air density (kg/m³)
- A = Rotor swept area (m²)
- Cp = Power coefficient (function of tip-speed ratio λ and pitch angle β)
- v(t) = Wind speed at time t

**Time Series Forecasting with LSTM:**

```
h_t = tanh(W_h × h_{t-1} + W_x × x_t + b_h)
c_t = f_t ⊙ c_{t-1} + i_t ⊙ tanh(W_c × [h_{t-1}, x_t] + b_c)
```

**Implementation:**
```python
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Wind speed time series
wind_speeds = np.array([...])  # Historical wind data
power_outputs = np.array([...])  # Corresponding power outputs

# Prepare sequences for LSTM
def create_sequences(data, lookback=24):
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i])
        y.append(data[i])
    return np.array(X), np.array(y)

# Build LSTM model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(24, 1)),
    LSTM(50),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')
```

### 2. Smart Grid Optimization

**Optimal Power Flow (OPF) Problem:**

```
Minimize: Σᵢ cᵢ(Pᵢ)
Subject to:
  Pᵢ^min ≤ Pᵢ ≤ Pᵢ^max
  Qᵢ^min ≤ Qᵢ ≤ Qᵢ^max
  |Vᵢ|^min ≤ |Vᵢ| ≤ |Vᵢ|^max
  Σᵢ Pᵢ = Σᵢ Pᵢ^load
```

Where:
- cᵢ(Pᵢ) = Cost function for generator i
- Pᵢ, Qᵢ = Active and reactive power
- Vᵢ = Voltage magnitude
- Pᵢ^load = Load demand

**Reinforcement Learning for Grid Control:**

```
Q(s,a) = Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
```

Where:
- s = Grid state (voltages, flows, loads)
- a = Control actions (generator dispatch, storage)
- r = Reward (cost reduction, stability)

### 3. Energy Storage Optimization

**Battery State of Charge (SOC) Model:**

```
SOC(t) = SOC(t-1) + (P_charge(t) × η_charge - P_discharge(t)/η_discharge) × Δt / Capacity
```

**Optimal Storage Dispatch:**

```
Maximize: Σᵗ (P_discharge(t) × Price(t) - P_charge(t) × Price(t))
Subject to:
  SOC_min ≤ SOC(t) ≤ SOC_max
  P_charge(t) ≤ P_charge_max
  P_discharge(t) ≤ P_discharge_max
```

---

## 💻 Implementation

### 1. Renewable Energy Forecasting System

```python
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

class RenewableEnergyForecaster:
    def __init__(self, forecast_horizon=24):
        self.forecast_horizon = forecast_horizon
        self.scaler = StandardScaler()
        self.model = self.build_lstm_model()
    
    def build_lstm_model(self):
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(24, 5)),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(self.forecast_horizon)
        ])
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        return model
    
    def prepare_data(self, data):
        """Prepare time series data for LSTM"""
        features = ['wind_speed', 'temperature', 'humidity', 'pressure', 'hour']
        
        # Create lagged features
        for i in range(1, 25):
            for feature in ['wind_speed', 'temperature']:
                data[f'{feature}_lag_{i}'] = data[feature].shift(i)
        
        # Create cyclical features for time
        data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
        data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)
        
        return data
    
    def create_sequences(self, data, target_col='power_output'):
        """Create sequences for LSTM training"""
        X, y = [], []
        
        for i in range(24, len(data) - self.forecast_horizon):
            # Input sequence (24 hours of features)
            sequence = data.iloc[i-24:i][['wind_speed', 'temperature', 'humidity', 'pressure', 'hour_sin']].values
            X.append(sequence)
            
            # Target sequence (next 24 hours of power output)
            target = data.iloc[i:i+self.forecast_horizon][target_col].values
            y.append(target)
        
        return np.array(X), np.array(y)
    
    def train(self, historical_data):
        """Train the forecasting model"""
        # Prepare data
        data = self.prepare_data(historical_data)
        X, y = self.create_sequences(data)
        
        # Split data
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=50,
            batch_size=32,
            verbose=1
        )
        
        return history
    
    def forecast(self, current_conditions):
        """Generate power forecast for next 24 hours"""
        # Prepare input sequence
        sequence = self.prepare_input_sequence(current_conditions)
        
        # Make prediction
        forecast = self.model.predict(sequence)
        
        return forecast[0]
    
    def prepare_input_sequence(self, conditions):
        """Prepare current conditions for forecasting"""
        # This would use the last 24 hours of data
        # Simplified for demonstration
        sequence = np.random.randn(1, 24, 5)  # Mock data
        return sequence

# Usage example
forecaster = RenewableEnergyForecaster()

# Simulate historical data
dates = pd.date_range('2025-01-01', periods=1000, freq='H')
historical_data = pd.DataFrame({
    'timestamp': dates,
    'wind_speed': np.random.uniform(5, 25, 1000),
    'temperature': np.random.uniform(10, 30, 1000),
    'humidity': np.random.uniform(40, 90, 1000),
    'pressure': np.random.uniform(1000, 1020, 1000),
    'hour': dates.hour,
    'power_output': np.random.uniform(0, 100, 1000)
})

# Train model
history = forecaster.train(historical_data)

# Generate forecast
current_conditions = {
    'wind_speed': 15.5,
    'temperature': 22.0,
    'humidity': 65.0,
    'pressure': 1013.0,
    'hour': 14
}

forecast = forecaster.forecast(current_conditions)
print(f"24-hour power forecast: {forecast}")
```

### 2. Smart Grid Optimization System

```python
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import cvxpy as cp

class SmartGridOptimizer:
    def __init__(self):
        self.generators = {}
        self.loads = {}
        self.storage_units = {}
        self.grid_constraints = {}
    
    def add_generator(self, name, min_power, max_power, cost_function):
        """Add a power generator to the grid"""
        self.generators[name] = {
            'min_power': min_power,
            'max_power': max_power,
            'cost_function': cost_function
        }
    
    def add_load(self, name, power_demand):
        """Add a load to the grid"""
        self.loads[name] = power_demand
    
    def add_storage(self, name, capacity, max_charge, max_discharge, efficiency):
        """Add energy storage to the grid"""
        self.storage_units[name] = {
            'capacity': capacity,
            'max_charge': max_charge,
            'max_discharge': max_discharge,
            'efficiency': efficiency
        }
    
    def optimize_dispatch(self, time_periods=24):
        """Optimize power dispatch for the entire grid"""
        # Decision variables
        generator_power = {}
        storage_charge = {}
        storage_discharge = {}
        storage_soc = {}
        
        # Initialize variables for all time periods
        for gen_name in self.generators:
            generator_power[gen_name] = cp.Variable(time_periods, nonneg=True)
        
        for storage_name in self.storage_units:
            storage_charge[storage_name] = cp.Variable(time_periods, nonneg=True)
            storage_discharge[storage_name] = cp.Variable(time_periods, nonneg=True)
            storage_soc[storage_name] = cp.Variable(time_periods, nonneg=True)
        
        # Objective function: minimize total cost
        total_cost = 0
        for gen_name, gen_data in self.generators.items():
            for t in range(time_periods):
                # Linear cost function: cost = a * power + b
                cost = gen_data['cost_function'](generator_power[gen_name][t])
                total_cost += cost
        
        # Constraints
        constraints = []
        
        # Power balance constraint
        for t in range(time_periods):
            total_generation = cp.sum([generator_power[gen][t] for gen in self.generators])
            total_load = sum(self.loads.values())  # Simplified: constant load
            total_storage_net = cp.sum([
                storage_discharge[storage][t] - storage_charge[storage][t]
                for storage in self.storage_units
            ])
            
            constraints.append(total_generation + total_storage_net == total_load)
        
        # Generator constraints
        for gen_name, gen_data in self.generators.items():
            for t in range(time_periods):
                constraints.append(generator_power[gen_name][t] >= gen_data['min_power'])
                constraints.append(generator_power[gen_name][t] <= gen_data['max_power'])
        
        # Storage constraints
        for storage_name, storage_data in self.storage_units.items():
            for t in range(time_periods):
                # SOC evolution
                if t == 0:
                    constraints.append(storage_soc[storage_name][t] == 
                                   storage_charge[storage_name][t] * storage_data['efficiency'] -
                                   storage_discharge[storage_name][t] / storage_data['efficiency'])
                else:
                    constraints.append(storage_soc[storage_name][t] == 
                                   storage_soc[storage_name][t-1] +
                                   storage_charge[storage_name][t] * storage_data['efficiency'] -
                                   storage_discharge[storage_name][t] / storage_data['efficiency'])
                
                # SOC bounds
                constraints.append(storage_soc[storage_name][t] >= 0)
                constraints.append(storage_soc[storage_name][t] <= storage_data['capacity'])
                
                # Charge/discharge limits
                constraints.append(storage_charge[storage_name][t] <= storage_data['max_charge'])
                constraints.append(storage_discharge[storage_name][t] <= storage_data['max_discharge'])
        
        # Solve optimization problem
        problem = cp.Problem(cp.Minimize(total_cost), constraints)
        problem.solve()
        
        return {
            'status': problem.status,
            'optimal_cost': problem.value,
            'generator_power': {gen: generator_power[gen].value for gen in self.generators},
            'storage_soc': {storage: storage_soc[storage].value for storage in self.storage_units}
        }

# Usage example
optimizer = SmartGridOptimizer()

# Add generators
optimizer.add_generator('solar_farm', 0, 100, lambda p: 0.05 * p)  # Solar: $0.05/kWh
optimizer.add_generator('wind_farm', 0, 80, lambda p: 0.08 * p)    # Wind: $0.08/kWh
optimizer.add_generator('gas_plant', 20, 200, lambda p: 0.12 * p)  # Gas: $0.12/kWh

# Add loads
optimizer.add_load('residential', 150)  # 150 MW demand

# Add storage
optimizer.add_storage('battery', capacity=50, max_charge=20, max_discharge=20, efficiency=0.9)

# Optimize dispatch
result = optimizer.optimize_dispatch(time_periods=24)

print(f"Optimization status: {result['status']}")
print(f"Optimal cost: ${result['optimal_cost']:.2f}")
```

### 3. Energy Demand Response System

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
import streamlit as st

class DemandResponseSystem:
    def __init__(self):
        self.demand_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.price_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.load_profiles = {}
    
    def generate_historical_data(self, days=365):
        """Generate synthetic historical demand and price data"""
        dates = pd.date_range('2025-01-01', periods=days*24, freq='H')
        
        # Base demand with daily and weekly patterns
        base_demand = 1000  # MW
        daily_pattern = np.sin(2 * np.pi * dates.hour / 24) * 200
        weekly_pattern = np.sin(2 * np.pi * dates.dayofweek / 7) * 50
        
        # Add weather effects
        temperature = np.random.uniform(10, 35, len(dates))
        weather_effect = (temperature - 20) * 10
        
        # Add random noise
        noise = np.random.normal(0, 30, len(dates))
        
        demand = base_demand + daily_pattern + weekly_pattern + weather_effect + noise
        demand = np.maximum(demand, 0)  # No negative demand
        
        # Generate price data (inverse relationship with demand)
        base_price = 50  # $/MWh
        price = base_price + (demand - base_demand) * 0.1 + np.random.normal(0, 5, len(dates))
        price = np.maximum(price, 10)  # Minimum price
        
        return pd.DataFrame({
            'timestamp': dates,
            'demand': demand,
            'price': price,
            'temperature': temperature,
            'hour': dates.hour,
            'day_of_week': dates.dayofweek,
            'month': dates.month
        })
    
    def train_demand_model(self, historical_data):
        """Train model to predict demand"""
        features = ['hour', 'day_of_week', 'month', 'temperature']
        X = historical_data[features]
        y = historical_data['demand']
        
        self.demand_model.fit(X, y)
    
    def train_price_model(self, historical_data):
        """Train model to predict electricity prices"""
        features = ['hour', 'day_of_week', 'month', 'temperature', 'demand']
        X = historical_data[features]
        y = historical_data['price']
        
        self.price_model.fit(X, y)
    
    def predict_demand(self, conditions):
        """Predict demand for given conditions"""
        features = np.array([[
            conditions['hour'],
            conditions['day_of_week'],
            conditions['month'],
            conditions['temperature']
        ]])
        
        return self.demand_model.predict(features)[0]
    
    def predict_price(self, conditions):
        """Predict electricity price for given conditions"""
        features = np.array([[
            conditions['hour'],
            conditions['day_of_week'],
            conditions['month'],
            conditions['temperature'],
            conditions['demand']
        ]])
        
        return self.price_model.predict(features)[0]
    
    def optimize_demand_response(self, current_conditions, flexible_loads):
        """Optimize demand response for flexible loads"""
        # Predict baseline demand and price
        baseline_demand = self.predict_demand(current_conditions)
        baseline_price = self.predict_price({
            **current_conditions,
            'demand': baseline_demand
        })
        
        # Optimize flexible loads
        optimized_schedule = {}
        total_cost_savings = 0
        
        for load_name, load_data in flexible_loads.items():
            # Find optimal time slots for this load
            best_slots = self.find_optimal_slots(
                load_data['duration'],
                load_data['power'],
                current_conditions,
                baseline_demand
            )
            
            optimized_schedule[load_name] = best_slots
            
            # Calculate cost savings
            original_cost = load_data['power'] * load_data['duration'] * baseline_price
            optimized_cost = self.calculate_optimized_cost(load_data, best_slots, current_conditions)
            total_cost_savings += original_cost - optimized_cost
        
        return {
            'baseline_demand': baseline_demand,
            'baseline_price': baseline_price,
            'optimized_schedule': optimized_schedule,
            'cost_savings': total_cost_savings
        }
    
    def find_optimal_slots(self, duration, power, conditions, baseline_demand):
        """Find optimal time slots for flexible load"""
        # Simple heuristic: find lowest price periods
        prices = []
        for hour in range(24):
            test_conditions = {**conditions, 'hour': hour}
            test_demand = self.predict_demand(test_conditions) + power
            price = self.predict_price({**test_conditions, 'demand': test_demand})
            prices.append((hour, price))
        
        # Sort by price and select lowest price slots
        prices.sort(key=lambda x: x[1])
        optimal_slots = [hour for hour, _ in prices[:duration]]
        
        return optimal_slots
    
    def calculate_optimized_cost(self, load_data, slots, conditions):
        """Calculate cost for optimized load schedule"""
        total_cost = 0
        for slot in slots:
            test_conditions = {**conditions, 'hour': slot}
            test_demand = self.predict_demand(test_conditions) + load_data['power']
            price = self.predict_price({**test_conditions, 'demand': test_demand})
            total_cost += load_data['power'] * price
        
        return total_cost

# Usage example
dr_system = DemandResponseSystem()

# Generate and train on historical data
historical_data = dr_system.generate_historical_data()
dr_system.train_demand_model(historical_data)
dr_system.train_price_model(historical_data)

# Define flexible loads
flexible_loads = {
    'ev_charging': {
        'power': 7.2,  # kW
        'duration': 4,  # hours
        'deadline': 8   # must complete by 8 AM
    },
    'dishwasher': {
        'power': 1.8,  # kW
        'duration': 2,  # hours
        'deadline': 24  # can run anytime
    },
    'laundry': {
        'power': 2.5,  # kW
        'duration': 3,  # hours
        'deadline': 20  # must complete by 8 PM
    }
}

# Current conditions
current_conditions = {
    'hour': 18,
    'day_of_week': 2,  # Wednesday
    'month': 6,
    'temperature': 25.0
}

# Optimize demand response
result = dr_system.optimize_demand_response(current_conditions, flexible_loads)

print(f"Baseline demand: {result['baseline_demand']:.1f} MW")
print(f"Baseline price: ${result['baseline_price']:.2f}/MWh")
print(f"Cost savings: ${result['cost_savings']:.2f}")
print("\nOptimized schedule:")
for load, slots in result['optimized_schedule'].items():
    print(f"{load}: {slots}")
```

---

## 🎯 Applications

### 1. Renewable Energy Integration

**Tesla's Virtual Power Plant:**
- 50,000+ homes with solar + Powerwall
- 250 MW virtual power plant
- Real-time grid services
- $2M+ revenue in 2024

**Case Study: California Grid**
- 40% renewable energy in 2024
- AI forecasting reduces curtailment by 30%
- Dynamic pricing reduces peak demand by 15%

### 2. Smart Grid Management

**Grid4C's Predictive Analytics:**
- 99.7% accuracy in demand forecasting
- 25% reduction in grid losses
- Real-time anomaly detection
- Automated load balancing

### 3. Energy Storage Optimization

**Tesla Megapack Applications:**
- 100 MWh battery installations
- 4-hour grid services
- Frequency regulation
- Peak shaving

### 4. Demand Response Programs

**PJM Interconnection:**
- 10,000+ MW demand response capacity
- Real-time price signals
- Automated load shedding
- $1B+ annual savings

---

## 🧪 Exercises and Projects

### Exercise 1: Solar Power Forecasting

**Task**: Build a model to predict solar power generation based on weather data.

**Dataset**: Use solar irradiance and weather data.

**Requirements**:
- 24-hour ahead forecasting
- RMSE < 15%
- Handle cloud cover effects

### Exercise 2: Grid Stability Analysis

**Task**: Analyze grid stability with increasing renewable penetration.

**Components**:
- Load flow analysis
- Frequency stability
- Voltage regulation
- Contingency analysis

### Exercise 3: Energy Storage Optimization

**Task**: Optimize battery dispatch for maximum revenue.

**Constraints**:
- State of charge limits
- Charge/discharge rates
- Cycle life considerations
- Market price volatility

### Project: Microgrid Energy Management System

**Objective**: Build a complete microgrid management system.

**Components**:
1. **Renewable Forecasting**: Solar and wind prediction
2. **Load Forecasting**: Demand prediction
3. **Storage Optimization**: Battery dispatch optimization
4. **Grid Integration**: Utility grid interaction
5. **Economic Analysis**: Cost-benefit optimization

**Implementation Steps**:
```python
# 1. Renewable forecasting
class RenewableForecaster:
    def forecast_solar(self, weather_data):
        # Predict solar generation
        pass
    
    def forecast_wind(self, weather_data):
        # Predict wind generation
        pass

# 2. Load forecasting
class LoadForecaster:
    def forecast_demand(self, historical_data, weather_data):
        # Predict electricity demand
        pass

# 3. Storage optimization
class StorageOptimizer:
    def optimize_dispatch(self, generation, demand, prices):
        # Optimize battery operation
        pass

# 4. Economic analysis
class EconomicAnalyzer:
    def calculate_roi(self, investment, savings):
        # Calculate return on investment
        pass
```

### Quiz Questions

1. **What is the primary challenge of renewable energy integration?**
   - A) High installation costs
   - B) Intermittency and variability
   - C) Limited technology
   - D) Regulatory barriers

2. **Which AI technique is most suitable for energy demand forecasting?**
   - A) Linear regression
   - B) Time series analysis
   - C) Clustering
   - D) Reinforcement learning

3. **What is the main benefit of smart grids?**
   - A) Lower electricity prices
   - B) Two-way communication and real-time optimization
   - C) Reduced maintenance
   - D) Faster installation

**Answers**: 1-B, 2-B, 3-B

---

## 📖 Further Reading

### Essential Papers
1. **"Deep Learning for Renewable Energy Forecasting"** - Zhang et al. (2019)
2. **"Smart Grid Optimization: A Survey"** - Momoh (2012)
3. **"Energy Storage in Power Systems"** - Strbac et al. (2016)

### Books
1. **"Smart Grids: Fundamentals and Technologies"** - Elsevier
2. **"Renewable Energy Integration"** - Academic Press
3. **"Energy Storage Systems"** - Springer

### Online Resources
1. **NREL Energy Data**: https://www.nrel.gov/data/
2. **IEA Energy Statistics**: https://www.iea.org/data-and-statistics
3. **Smart Grid Information**: https://www.smartgrid.gov/

### Next Steps
1. **Advanced Topics**: Explore grid cybersecurity
2. **Related Modules**: 
   - [Time Series Forecasting](specialized_ml/12_time_series_forecasting.md)
   - [Reinforcement Learning](core_ml_fields/11_rl_advanced.md)
   - [Sustainability AI](advanced_topics/54_ai_sustainability.md)

---

## 🎯 Key Takeaways

1. **Renewable Forecasting**: AI enables accurate prediction of variable renewable energy
2. **Grid Optimization**: Smart grids use AI for real-time optimization and stability
3. **Energy Storage**: AI optimizes battery dispatch for maximum value
4. **Demand Response**: AI enables intelligent load shifting and peak shaving
5. **Economic Benefits**: AI-driven energy systems reduce costs and improve efficiency
6. **Sustainability**: AI accelerates the transition to clean energy systems

---

*"The future of energy is not just renewable, but intelligent."*

**Next: [AI in Transportation](78_ai_in_transportation/README.md) → Autonomous vehicles and smart mobility systems** 