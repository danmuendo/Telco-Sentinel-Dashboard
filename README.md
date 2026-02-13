# 🔋 Telco-Sentinel: Predictive Maintenance for Telecom Infrastructure

### **Project Overview**
**Telco-Sentinel** is a data-driven "Digital Twin" application designed to optimize the maintenance of battery backup systems at remote Telecommunications Base Transceiver Stations (BTS). 

Using the NASA Li-ion Battery Aging Dataset, this tool bridges the gap between raw hardware sensor logs and actionable business intelligence. It allows infrastructure managers to move from **Reactive Maintenance** (replacing failed parts) to **Predictive Intelligence** (forecasting failure before it happens).

---

## 🚀 Key Features

* **State of Health (SoH) Monitoring:** Real-time calculation of battery health based on capacity fade and electrolyte resistance.
* **RUL Forecasting:** Implements a Linear Regression model to predict the **Remaining Useful Life** (cycles) before an asset hits the critical 1.4Ah failure threshold.
* **Digital Twin Deep-Dive:** An interactive sensor-level explorer that allows engineers to visualize Voltage, Current, and Temperature profiles for any specific historical cycle.
* **Maintenance Alerting:** Automatic color-coded status indicators (Healthy, Replace Soon, Critical) based on predictive trends.

---

## 🛠️ Tech Stack

* **Language:** Python 3.x
* **Frontend/Dashboard:** Streamlit
* **Data Analysis:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn (Linear Regression)
* **Visualization:** Plotly (Interactive Graphs)

---

## 📂 Project Structure

```text
/Tetranet_Project
├── app.py              # Main Streamlit application logic
├── metadata.csv        # Global index of site logs and capacity metrics
├── requirements.txt    # Python dependencies for deployment
├── .gitignore          # Prevents heavy data/environment files from being tracked
└── /data               # (Local Only) Contains detailed sensor CSV logs (7,000+ files)

# ⚙️ Installation & Usage
Clone the repository:
