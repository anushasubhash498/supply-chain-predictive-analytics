import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Set style
sns.set_theme(style="darkgrid")
plt.rcParams['figure.figsize'] = (10, 6)

# Paths
base_dir = r'C:\Users\anusu\.gemini\antigravity\scratch\analytics-portfolio\supply-chain-predictive-analytics'
data_path = os.path.join(base_dir, 'data', 'inventory_logs.csv')
output_dir = os.path.join(base_dir, 'outputs')
os.makedirs(output_dir, exist_ok=True)

if not os.path.exists(data_path):
    raise FileNotFoundError(f"Data file not found at {data_path}. Run generate_data.py first.")

df = pd.read_csv(data_path)
df['date'] = pd.to_datetime(df['date'])

print("=== Supply Chain Dataset Profile ===")
print(f"Total rows: {len(df)}")
print(f"Products monitored: {df['product_name'].unique()}")

# Calculate Safety Stock and Reorder Points
# Formulas:
# Safety Stock = (Max Sales * Max Lead Time) - (Avg Sales * Avg Lead Time)
# Reorder Point = (Avg Sales * Avg Lead Time) + Safety Stock
# Here lead time variance is modeled as lead_time * 1.5 for Max

inventory_params = []
for pid in df['product_id'].unique():
    p_df = df[df['product_id'] == pid]
    
    avg_sales = p_df['sales_qty'].mean()
    max_sales = p_df['sales_qty'].max()
    
    avg_lt = p_df['lead_time_days'].iloc[0]
    max_lt = avg_lt + 2 # simulated max lead time
    
    safety_stock = int((max_sales * max_lt) - (avg_sales * avg_lt))
    reorder_point = int((avg_sales * avg_lt) + safety_stock)
    
    total_sales = p_df['sales_qty'].sum()
    stockout_events = p_df['stockout_event'].sum()
    
    inventory_params.append({
        'product_id': pid,
        'product_name': p_df['product_name'].iloc[0],
        'avg_daily_sales': round(avg_sales, 1),
        'safety_stock': safety_stock,
        'reorder_point': reorder_point,
        'stockout_events': stockout_events,
        'revenue': round(total_sales * p_df['unit_price'].iloc[0], 2)
    })

params_df = pd.DataFrame(inventory_params)
print("\n=== Inventory Optimization Parameters ===")
print(params_df.to_string(index=False))

# 1. Plot Reorder Points and Safety Stock comparison
plt.figure(figsize=(10, 6))
bar_width = 0.35
index = np.arange(len(params_df))
plt.bar(index, params_df['reorder_point'], bar_width, label='Reorder Point', color='royalblue')
plt.bar(index + bar_width, params_df['safety_stock'], bar_width, label='Safety Stock', color='salmon')
plt.xlabel('Product', fontsize=12)
plt.ylabel('Units', fontsize=12)
plt.title('Optimal Reorder Points vs Safety Stock Levels by Product', fontsize=14, fontweight='bold')
plt.xticks(index + bar_width / 2, params_df['product_name'], rotation=30, ha='right')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'inventory_parameters.png'), dpi=300)
plt.close()

# 2. Daily Stock Levels & Safety Stock over time for P001
p1_df = df[df['product_id'] == 'P001'].sort_values('date').tail(90) # Last 90 days
p1_params = params_df[params_df['product_id'] == 'P001'].iloc[0]

plt.figure(figsize=(12, 6))
plt.plot(p1_df['date'], p1_df['stock_level'], label='Current Stock Level', color='teal', linewidth=2)
plt.axhline(y=p1_params['reorder_point'], color='orange', linestyle='--', label=f'Reorder Point ({p1_params["reorder_point"]})')
plt.axhline(y=p1_params['safety_stock'], color='red', linestyle=':', label=f'Safety Stock ({p1_params["safety_stock"]})')
plt.title(f'Stock Level Trend & Triggers for {p1_params["product_name"]} (Last 90 Days)', fontsize=14, fontweight='bold')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Stock Level (Units)', fontsize=12)
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'stock_level_trend.png'), dpi=300)
plt.close()

# 3. Demand Forecasting ML Model (Random Forest)
# We will forecast sales of Bio-Coffee Beans (P001)
p1_full = df[df['product_id'] == 'P001'].sort_values('date').copy()

# Feature engineering: create lags
for lag in [1, 2, 3, 7, 14]:
    p1_full[f'sales_lag_{lag}'] = p1_full['sales_qty'].shift(lag)
    
p1_full['rolling_mean_7'] = p1_full['sales_qty'].shift(1).rolling(window=7).mean()
p1_full['day_of_week'] = p1_full['date'].dt.weekday

# Drop rows with NaN due to shift
p1_full = p1_full.dropna()

# Split into train & test (last 30 days as test)
train = p1_full.iloc[:-30]
test = p1_full.iloc[-30:]

features = ['sales_lag_1', 'sales_lag_2', 'sales_lag_3', 'sales_lag_7', 'sales_lag_14', 'rolling_mean_7', 'day_of_week']
target = 'sales_qty'

X_train, y_train = train[features], train[target]
X_test, y_test = test[features], test[target]

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict
test['predictions'] = model.predict(X_test)

mse = mean_squared_error(y_test, test['predictions'])
mae = mean_absolute_error(y_test, test['predictions'])

print("\n=== Machine Learning Demand Forecasting ===")
print(f"Model: RandomForestRegressor")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")

# Plot Actual vs Predicted Demand
plt.figure(figsize=(12, 6))
plt.plot(test['date'], y_test, label='Actual Demand', color='darkblue', marker='o', linewidth=2)
plt.plot(test['date'], test['predictions'], label='Forecasted Demand', color='crimson', linestyle='--', marker='x', linewidth=2)
plt.title(f'15-Day Demand Forecast vs Actuals for {p1_params["product_name"]}', fontsize=14, fontweight='bold')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Sales Volume (Units)', fontsize=12)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'demand_forecast.png'), dpi=300)
plt.close()

print(f"\nCharts and figures saved to {output_dir}")
