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

## Google Drive Assets

👉 [Google Drive Link](https://drive.google.com/drive/folders/1xOgzf7MHQS7ABuCT3045WyAxbP_IBaX9?usp=sharing)
