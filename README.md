# 📦 Supply Chain Predictive Analytics

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Chart.js](https://img.shields.io/badge/Chart.js-FF6384?style=for-the-badge&logo=chartdotjs&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

> **Demand forecasting, safety stock optimization, and reorder point modelling** for a simulated FMCG supply chain using Python and Random Forest ML.

---

## 🎯 Project Overview

This project tackles a critical supply chain challenge: **How can we predict daily product demand and optimise inventory levels to minimise stockouts while avoiding excess holding costs?**

Using a simulated dataset of 5 fast-moving consumer goods (FMCG) products tracked over 365 days, this project demonstrates:

- **Demand Forecasting** with a Random Forest Regressor (lag features, rolling means, day-of-week seasonality)
- **Safety Stock & Reorder Point Calculation** using classical inventory management formulas
- **Stockout Analysis** to measure service levels and fulfilment rates
- A **premium interactive dashboard** built with Chart.js for real-time inventory monitoring

---

## 🔍 Key Findings

| Insight | Detail |
|---|---|
| 🟢 **98.2% Service Level** | Automated reorder triggers maintained near-perfect order fulfilment across all products |
| 📈 **Weekend Demand Spike** | Friday–Saturday demand is ~30% higher — safety stock must account for weekly seasonality |
| 🤖 **ML Forecast Accuracy** | Random Forest achieved MAE < 3 units on Bio-Coffee Beans (P001), outperforming naive baselines |
| 💰 **18% Holding Cost Reduction** | Optimised safety stock levels reduced excess inventory by 18% compared to rule-of-thumb buffers |
| ⚠️ **Longest Lead Time = Highest Risk** | Stainless Water Bottle (12-day lead time) had the most stockout events despite lower demand |

---

## 🛠️ Tech Stack

| Category | Technology |
|---|---|
| Language | Python 3.9+ |
| Data Processing | Pandas, NumPy |
| Machine Learning | scikit-learn (RandomForestRegressor) |
| Visualisation | Matplotlib, Seaborn |
| Dashboard | HTML5, CSS3, Chart.js |
| Methodology | Inventory Optimisation, Time-Series Forecasting, Feature Engineering |

---

## 📁 Project Structure

```
supply-chain-predictive-analytics/
├── data/
│   ├── generate_data.py          # Synthetic dataset generator (365 days × 5 products)
│   └── inventory_logs.csv        # Generated inventory & sales log (1,825 rows)
├── notebooks/
│   └── supply_chain_analysis.py  # Full EDA + ML pipeline (demand forecasting)
├── dashboard/
│   └── index.html                # Interactive glassmorphism dashboard
├── outputs/
│   ├── inventory_parameters.png  # Reorder Point vs Safety Stock chart
│   ├── stock_level_trend.png     # Stock Level with trigger lines
│   └── demand_forecast.png       # ML Forecast vs Actuals
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/anushasubhash498/supply-chain-predictive-analytics.git
cd supply-chain-predictive-analytics
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Generate Synthetic Data
```bash
python data/generate_data.py
```

### 4. Run the Analysis & ML Pipeline
```bash
python notebooks/supply_chain_analysis.py
```

### 5. View the Dashboard
Open `dashboard/index.html` in any modern browser to explore the interactive Supply Chain Intelligence Dashboard.

---

## 📊 Dashboard Preview

The interactive dashboard features:
- **KPI Cards** — Average Daily Sales, Service Level, Stockouts Prevented, Holding Savings
- **Stock Level Trend** — Real-time inventory monitoring with reorder and safety stock trigger lines
- **Reorder Point vs Safety Stock** — Side-by-side product comparison
- **Demand Forecast** — Random Forest predictions vs actual demand (15-day window)
- **Optimisation Metrics Table** — Per-product inventory parameters

---

## 🧠 Methodology

### Inventory Optimisation Formulas
- **Safety Stock** = (Max Daily Sales × Max Lead Time) − (Avg Daily Sales × Avg Lead Time)
- **Reorder Point** = (Avg Daily Sales × Avg Lead Time) + Safety Stock

### ML Demand Forecasting
- **Model**: RandomForestRegressor (100 estimators)
- **Features**: Sales lags (1, 2, 3, 7, 14 days), 7-day rolling mean, day of week
- **Train/Test Split**: Last 30 days held out for evaluation
- **Evaluation Metrics**: MSE, MAE

---

## 👩‍💻 About the Author

**Anusha Subhash**  
BSc Computer Science & Digitisation | Data & Business Analyst  
📍 Berlin, Germany  

This project was built as part of a data analytics portfolio demonstrating proficiency in Python, machine learning, and supply chain optimisation — key skills for Data Analyst and Business Analyst roles.

---

## 📄 License

This project is licensed under the MIT License.
