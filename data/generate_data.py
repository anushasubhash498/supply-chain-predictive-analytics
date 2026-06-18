import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

# Set seed for reproducibility
np.random.seed(42)

# Parameters
n_days = 365
start_date = datetime.now() - timedelta(days=n_days)

# 5 Products with different average demand and lead times
products_info = [
    {'id': 'P001', 'name': 'Bio-Coffee Beans 1kg', 'avg_demand': 25, 'lead_time': 5, 'cost': 12.50, 'price': 24.99},
    {'id': 'P002', 'name': 'Oat Milk Premium 1L', 'avg_demand': 60, 'lead_time': 3, 'cost': 0.85, 'price': 1.99},
    {'id': 'P003', 'name': 'Eco Dishwasher Tabs', 'avg_demand': 15, 'lead_time': 7, 'cost': 4.20, 'price': 8.99},
    {'id': 'P004', 'name': 'Stainless Water Bottle', 'avg_demand': 8, 'lead_time': 12, 'cost': 6.50, 'price': 19.99},
    {'id': 'P005', 'name': 'Organic Avocado Oil', 'avg_demand': 12, 'lead_time': 6, 'cost': 5.80, 'price': 12.49}
]

data = []
for p in products_info:
    # Initialize inventory
    stock = p['avg_demand'] * (p['lead_time'] + 3) # Starting stock
    reorder_point = p['avg_demand'] * p['lead_time']
    order_placed = False
    days_until_arrival = 0
    order_qty = p['avg_demand'] * 15 # Order 15 days worth of stock
    
    for day in range(n_days):
        current_date = start_date + timedelta(days=day)
        
        # Seasonality: higher sales on weekends (Friday=4, Saturday=5)
        weekday = current_date.weekday()
        season_mult = 1.3 if weekday in [4, 5] else 0.9
        
        # Daily sales with poisson distribution
        sales = np.random.poisson(p['avg_demand'] * season_mult)
        
        # Handle stockouts
        if sales > stock:
            sales_realized = stock
            stock = 0
            stockout = 1
        else:
            sales_realized = sales
            stock -= sales
            stockout = 0
            
        # Inventory replenishment logic
        if order_placed:
            days_until_arrival -= 1
            if days_until_arrival <= 0:
                stock += order_qty
                order_placed = False
                
        # Reorder trigger
        if stock <= reorder_point and not order_placed:
            order_placed = True
            # Lead time varies slightly around avg lead time
            days_until_arrival = int(np.random.normal(p['lead_time'], 1.0))
            if days_until_arrival < 1:
                days_until_arrival = 1
                
        data.append([
            current_date.strftime('%Y-%m-%d'), p['id'], p['name'], 
            sales_realized, stock, p['lead_time'], days_until_arrival if order_placed else 0,
            p['cost'], p['price'], stockout
        ])

columns = [
    'date', 'product_id', 'product_name', 'sales_qty', 'stock_level', 
    'lead_time_days', 'days_until_replenishment', 'cost_price', 'unit_price', 'stockout_event'
]
df = pd.DataFrame(data, columns=columns)

# Save to CSV
base_dir = r'C:\Users\anusu\.gemini\antigravity\scratch\analytics-portfolio\supply-chain-predictive-analytics\data'
os.makedirs(base_dir, exist_ok=True)
output_path = os.path.join(base_dir, 'inventory_logs.csv')
df.to_csv(output_path, index=False)

print(f"Generated Inventory/Sales daily log dataset with {len(df)} records at {output_path}")
