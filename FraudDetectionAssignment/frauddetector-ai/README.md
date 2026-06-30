# 🕵️‍♀️ FraudDetect-AI: A Hybrid ML + Rule-Based Fraud Detection System

**FraudDetect-AI** is a real-time fraud detection system that combines machine learning and rule-based logic to flag suspicious transactions. It features a Streamlit dashboard, explainable AI (SHAP), and a modular backend — all crafted for production-grade clarity and interview showcase readiness.

---

## 🚀 Demo

> _Coming soon_: Link to Streamlit Cloud demo  
> _Local setup instructions below_

---

## 🎯 Features

- ✅ ML models (XGBoost, Isolation Forest)
- ✅ SMOTE for class imbalance
- ✅ SHAP visual explainability
- ✅ Rule-based fraud flagging (thresholds, transaction gaps)
- ✅ Streamlit dashboard with charts & filters
- ✅ Modular Python backend

---

## 📁 Project Structure

```
frauddetect-ai/
├── notebooks/            # Jupyter notebooks for EDA, modeling, SHAP
├── src/                  # Clean, modular Python pipeline
├── streamlit_app/        # Streamlit app UI
├── models/               # Trained models (.pkl)
├── data/                 # Raw dataset (gitignored)
├── docs/                 # Report and presentation
├── .gitignore
├── .gitattributes
├── README.md
├── requirements.txt
├── CONTRIBUTIONS.md
```

---

## 🔧 Local Setup

### 1. Clone the Repository

```
git clone https://github.com/<your-username>/frauddetect-ai.git
cd frauddetect-ai
```

### 2. Create a Virtual Environment

```
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install Requirements

```
pip install -r requirements.txt
```

### 4. Run the Streamlit App

```
streamlit run streamlit_app/app.py
```

### 5. Running Tests

To run the test suite:
```bash
pytest
```

<!-- force spacing -->

---

## 📊 Sample Output Screens

> Screenshots and visualizations will be added after the dashboard is finalized.  
> Place images in `docs/screenshots/` and reference them here.

---

## 🧠 Tech Stack

| Layer              | Technologies Used |
|-------------------|-------------------|
| ML Models          | `xgboost`, `scikit-learn`, `imbalanced-learn` |
| Explainability     | `SHAP` |
| Frontend Dashboard | `Streamlit`, `Plotly` |
| Visualization      | `matplotlib`, `seaborn` |
| Utilities          | `joblib`, `os`, `argparse`, `logging` |
| Dev Tools          | `PyCharm`, `Git`, `GitHub`, `Markdown` |

---

## 👥 Team & Contributions

Refer to [`CONTRIBUTIONS.md`](docs/CONTRIBUTIONS.md) for detailed breakdown of individual roles and work split.

---

## 📚 Academic Context

This project was developed as part of a college-level **Data Science & AI/ML major course**, focused on applying machine learning to solve real-world financial fraud detection challenges. It aims to meet industry expectations for model interpretability, clean code practices, and app integration.

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).  
Feel free to use, modify, and share — with proper attribution.

---

## ⭐ Support This Project

If you find this project helpful:
- ⭐ Star it on GitHub
- 🔁 Share with others
- 🍴 Fork and build upon it

Let’s fight fraud with clean code and smarter AI 
