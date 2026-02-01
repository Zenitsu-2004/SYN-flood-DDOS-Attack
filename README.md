# SYN Flood DDoS Attack Detection System Using Machine Learning

## Project Overview
This project implements an intelligent system to detect **SYN Flood DDoS attacks** using **Machine Learning and rule-based validation**.  
The system analyzes network flow data, predicts potential attacks using a **Random Forest model**, logs results into a **SQLite database**, and visualizes them through a **real-time web dashboard**.

---

## Technologies Used
- Python  
- Flask (Web Framework)  
- Scikit-learn (Machine Learning)  
- Pandas, NumPy (Data Processing)  
- SQLite (Database)  
- HTML, CSS (Frontend)  
- Chart.js / Plotly (Visualization)

---

## System Components
- **ML Detection Engine** – Random Forest classifier
- **Rule-Based Engine** – Validates SYN/ACK behavior
- **Threshold Logic** – Business optimization (Threshold = 0.4)
- **CSV Logger** – Stores predictions
- **SQLite Database** – Persistent logging
- **Flask Web App** – User interface
- **Dashboard** – KPIs, charts, logs table
- **Export Module** – PDF & CSV report

---

## Project Structure
```text
PROJECT/
│
├── data/ # CSV logs
├── models/ # Trained ML model
├── src/
│ ├── app.py # Main Flask app
│ ├── run_pipeline.py # Detection logic
│ ├── create_database.py
│ ├── csv_to_db.py
│ └── templates/
│ └── index.html # Web UI
│
├── ddos.db # SQLite database
├── README.md
```

---

## How to Run the Project

### Step 1: Activate Virtual Environment
```bash
.\.venv\Scripts\activate

### Step 2: Install Dependencies
pip install flask pandas joblib scikit-learn fpdf

### Step 3: Run the Web Application
python src/app.py

### Step 4: Open in Browser
http://127.0.0.1:5000

---

## Features

Real-time SYN flood detection
ML + Rule-based hybrid system
KPI dashboard
Attack vs Benign charts
Database logging
Export to PDF & CSV
Professional web interface

---

## Outputs & Results

Detection input/output screenshots
KPI cards (Total checks, Attacks, Benign)
Line graph: Threat score trend
Pie chart: Attack vs Benign
Bar chart: Cumulative trend
SQLite logs table

---

## Google Drive Assets

👉 [Google Drive Link](https://drive.google.com/drive/folders/1xOgzf7MHQS7ABuCT3045WyAxbP_IBaX9?usp=sharing)

## Conclusion
This system successfully demonstrates how Machine Learning combined with rule-based logic can effectively detect SYN flood DDoS attacks.
The integration of database logging and real-time dashboard makes the solution practical and suitable for real-world cybersecurity applications.

