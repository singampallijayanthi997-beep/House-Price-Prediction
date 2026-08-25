# 🏠 House Price Prediction using XGBoost

A Machine Learning web application that predicts the selling price of a house based on its features. The project uses **XGBoost Regression** for prediction and **Flask** for the web interface. It is deployed on **Render** for public access.

## 🌐 Live Demo

**Render:** https://house-price-prediction-czfa.onrender.com

---

## 📌 Project Overview

This application estimates the selling price of a house by analyzing important property features such as:

- Overall Quality
- Living Area
- Garage Capacity
- Basement Area
- First Floor Area
- Number of Bathrooms
- Total Rooms
- Year Built
- Neighborhood
- Kitchen Quality

The trained XGBoost model processes the input and returns the predicted house price in Indian currency format.

---

## ✨ Features

- Modern responsive web interface
- Real-time house price prediction
- XGBoost Regression model
- One-Hot Encoding for categorical features
- Indian currency formatting (₹)
- Input validation
- Deployed on Render
- Version controlled using Git & GitHub

---

## 🛠️ Tech Stack

| Technology | Purpose |
|------------|---------|
| Python | Programming Language |
| Flask | Web Framework |
| XGBoost | Machine Learning Model |
| Pandas | Data Processing |
| NumPy | Numerical Operations |
| Scikit-learn | Preprocessing |
| HTML/CSS | Frontend |
| Render | Cloud Deployment |
| Git & GitHub | Version Control |

---

## 📂 Project Structure

```text
House-Price-Prediction/
│
├── dataset/
│   └── housing.csv
│
├── model/
│   ├── xgboost_house_price_model.json
│   ├── one_hot_encoder.pkl
│   └── preprocessing_info.pkl
│
├── src/
│   ├── app.py
│   ├── predict.py
│   └── data_analysis.py
│
├── templates/
│   └── index.html
│
├── requirements.txt
├── runtime.txt
└── README.md
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/singampallijayanthi997-beep/House-Price-Prediction.git
cd House-Price-Prediction
```

### 2. Create virtual environment

```bash
python -m venv venv
```

### 3. Activate environment

**Windows**

```bash
venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the application

```bash
python src/app.py
```

Open:

```text
http://127.0.0.1:5000
```

---

## 🤖 Machine Learning Model

- **Algorithm:** XGBoost Regressor
- **Problem Type:** Regression
- **Target Variable:** House Sale Price
- **Preprocessing:** One-Hot Encoding + Feature Alignment

### Model Performance

| Metric | Value |
|---------|------:|
| R² Score | **91.23%** |
| MAE | **₹15,730.89** |
| RMSE | **₹25,942.49** |

---

## 📊 Prediction Workflow

```text
House Details
      │
      ▼
Data Preprocessing
      │
      ▼
One-Hot Encoding
      │
      ▼
Feature Alignment
      │
      ▼
XGBoost Regression
      │
      ▼
Predicted House Price
```

---

## 📸 Application Screenshots

- Home page with responsive input form
- House price prediction result
- Key house factors display
- Model performance metrics

---

## 👩‍💻 Author

**Singampalli Jayanthi**

- GitHub: https://github.com/singampallijayanthi997-beep

---

## 📄 License

This project is developed for educational and academic purposes.
