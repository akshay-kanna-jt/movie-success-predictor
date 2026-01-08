# 🎬 Movie Success Predictor 

A Machine Learning project that predicts the **success of a movie** based on IMDb metadata such as genres, ratings, cast, crew, and release details.

---

## 📌 Project Overview

The **Movie Success Predictor** uses historical movie data from IMDb to train a machine learning model that estimates whether a movie is likely to be successful or not.
This project demonstrates **data preprocessing, feature engineering, model training, and prediction**.

---

## 🛠️ Technologies Used

* **Python**
* **Pandas, NumPy**
* **Scikit-learn**
* **Flask** (for API / application layer)
* **IMDb Official Datasets**

---

## 📂 Project Structure

```
movie_success_predictor/
│
├── app.py                 # Main application
├── train_model.py         # Model training script
├── predict.py             # Prediction logic
├── requirements.txt       # Project dependencies
├── README.md              # Project documentation
│
├── data/                  # IMDb datasets (not included)
├── models/                # Trained models (generated locally)
└── venv/                  # Virtual environment (ignored)
```

---

## 📊 Dataset Information

Due to GitHub’s **file size limits**, large datasets are **not included** in this repository.

### 🔗 Download IMDb datasets from the official source:

👉 [https://datasets.imdbws.com/](https://datasets.imdbws.com/)

### Required files:

* `name.basics.tsv.gz`
* `title.basics.tsv.gz`
* `title.akas.tsv.gz`
* `title.crew.tsv.gz`
* `title.principals.tsv.gz`
* `title.ratings.tsv.gz`

### Setup:

1. Download the required files
2. Extract `.tsv.gz` files
3. Place them inside the `data/` directory

---

## 🧠 Model Information

Trained model files (`.pkl`) are excluded from GitHub because they exceed file size limits.

### To generate the model locally:

```bash
python train_model.py
```

This will train the model and save it in the project directory.

---

## ▶️ How to Run the Project

### 1️⃣ Clone the repository

```bash
git clone https://github.com/akshay-kanna-jt/movie-success-predictor.git
cd movie-success-predictor
```

### 2️⃣ Create and activate virtual environment

```bash
python -m venv venv
venv\Scripts\activate
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Train the model

```bash
python train_model.py
```

### 5️⃣ Run the application

```bash
python app.py
```

---

## 🎯 Use Cases

* Movie production planning
* OTT platform analysis
* Market trend prediction
* Machine learning academic projects

---

## 🚀 Future Enhancements

* Deep learning–based prediction
* Additional IMDb features
* Web UI dashboard
* Cloud deployment

---

## 👨‍💻 Author

**J T Akshay Kanna**
Aspiring Full Stack & Machine Learning Developer

---

## ⭐ Support

If you find this project useful, feel free to **star ⭐ the repository**.

---
