# Rock Drillability Index (DRI) Prediction Pipeline ⛏️

An end-to-end Machine Learning web application predicting the **Drilling Rate Index (DRI)** from geotechnical laboratory rock test parameters using Support Vector Regression (SVR).

🌐 **Live Web Application:** [Streamlit Dashboard](https://nirikshakotian-rock-drillability-ml-app-k2cpt4.streamlit.app)

---

## 📊 Model Performance

| Model | $R^2$ Score |
|---|---|
| **Tuned SVR (Scaled)** | **0.9823** |
| Linear Regression | 0.9816 |
| XGBoost | 0.9749 |
| Random Forest | 0.9735 |

---

## 🛠️ Tech Stack & Dependencies
* **Language:** Python 3.12
* **Machine Learning:** `scikit-learn` (`StandardScaler` + `SVR` RBF kernel)
* **Web UI:** Streamlit
* **Data Processing & Viz:** `pandas`, `numpy`, `matplotlib`, `seaborn`
* **Serialization:** `joblib`

---

## 🚀 How to Run Locally

1. Clone the repository:
   ```bash
   git clone [https://github.com/NirikshaKotian/Rock-Drillability-ML.git](https://github.com/NirikshaKotian/Rock-Drillability-ML.git)
   cd Rock-Drillability-ML