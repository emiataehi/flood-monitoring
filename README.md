# Real-Time Flood Anomaly Detection System

This project presents a machine learning-driven flood monitoring system designed to detect and predict flood anomalies across three monitoring stations in Greater Manchester — Bury Ground (River Irwell), Manchester Racecourse (River Irwell), and Rochdale (River Roch).

The system integrates real-time river level, rainfall, and meteorological data to detect potential flood risks using a hybrid ensemble approach that combines statistical analysis, Random Forest regression, and LSTM neural networks. It achieves over 97% accuracy in anomaly detection with sub-second response times, enabling near real-time alerts and decision support.

## 🌍 Live Dashboard

Interactive Streamlit App:https://floodmonitoringdashboardpy-w5atucr3y6mnjqzbwyfmyn.streamlit.app/

## ⚙️ System Overview

The system continuously collects and processes data from multiple sources: 

- Real-time river and rainfall data from UK Environment Agency APIs (updated every 15 minutes)
- Historical hydrological and meteorological datasets for baseline modelling

***Supabase cloud database for real-time data ingestion, storage, and retrieval***

**Data is cleaned, standardized, and analyzed through a pipeline that includes:**

- Missing data handling (Random Forest imputation & mode imputation)
- Feature engineering (rolling statistics, lag features, rate-of-change)
- Z-score & interquartile range analysis for statistical anomaly detection
- ML-based anomaly prediction via Random Forest and LSTM models
- Ensemble weighting for adaptive anomaly scoring

The Streamlit dashboard visualizes live and historical trends, anomalies, and alert thresholds, while the backend processes new data with an average latency of 525ms.

## 🧠 Machine Learning Approach

| Model                   | Purpose                                | Key Metrics                                         |
| ----------------------- | -------------------------------------- | --------------------------------------------------- |
| Random Forest Regressor | Detects abnormal river level patterns  | R² = 0.96                                           |
| LSTM Neural Network     | Predicts sequential water level trends | RMSE < 0.5                                          |
| Ensemble Method         | Combines Random Forest + LSTM outputs  | Accuracy = 97.7%, Precision = 95.5%, Recall = 94.9% |


Each station has its own trained models and dynamic thresholds to ensure local calibration of flood risk alerts.

## 🏗️ Architecture

**Tech Stack:**

- Python – Data processing, ML model training
- Streamlit – Interactive web dashboard
- Supabase (PostgreSQL) – Cloud-based real-time database
- Plotly & Matplotlib – Visualization
- Scikit-learn, TensorFlow/Keras – ML model development
- GitHub Actions – Automated updates and deployment

**Pipeline Workflow:**

1. Data ingestion → 2. Data preprocessing → 3. Feature extraction →
2. Model inference (Random Forest & LSTM) → 5. Anomaly scoring →
3. Visualization & alert generation

**📊 Example Dashboard Views**

- River level trends with anomaly highlights
- Real-time rainfall vs. flow correlation plots
- Risk-level indicators by station
- Time-lag correlation visualizations between upstream and downstream stations


## 🚀 Performance Summary

- Accuracy: 97.7%
- Precision: 95.5%
- Recall: 94.9%
- Average processing time: 525ms
- Alert delivery time: <5 seconds

## 📁 Project Structure
```
flood-monitoring/
│
├── data/                     # Historical and sample real-time data
├── notebooks/                # Model training and analysis notebooks
├── src/                      # Main source code for data pipeline & models
│   ├── data_processing.py
│   ├── anomaly_detection.py
│   ├── models/
│   ├── visualization/
│   └── config/
├── dashboard/                # Streamlit app files
├── requirements.txt
└── README.md
```

## 🔧 How to Run Locally

1. Clone the repository

git clone git clone https://github.com/emiataehi/flood-monitoring.git

cd flood-monitoring


2. Install dependencies

pip install -r requirements.txt


3. Run the Streamlit dashboard

streamlit run dashboard/app.py

(Optional) Configure your Supabase credentials in .env

## 🔍 Future Enhancements

- Expansion to additional UK monitoring stations
- Integration of adaptive learning for evolving flood dynamics
- Enhanced alerting via mobile notifications and email
- Deeper spatial correlation mapping using GIS data

## 📫 Contact

**For feedback, collaboration, or inquiries:**
Email: emi.igein@gmail.com

GitHub: [github.com/emiataehi](https://github.com/emiataehi/)

LinkedIn: [LinkedIn](https://www.linkedin.com/in/emi-igein-b024-8147)









