# 🌸 Iris Species Classifier

A machine learning project that classifies Iris flowers into **Setosa**, **Versicolor**, or **Virginica** based on sepal and petal measurements — built with exploratory data analysis, feature scaling, and hyperparameter-tuned classifiers (KNN & SVM).


🔗 **Live Demo:** [Add your deployed demo link here](https://your-demo-link.com) *(e.g. Streamlit Cloud / Hugging Face Spaces / Render)*

---

## 📖 Overview

This project uses the classic **Iris dataset** (150 samples, 4 features, 3 balanced classes) to build a supervised classification model. It walks through data inspection, visualization, preprocessing, model training, and hyperparameter tuning — then saves the final trained model for reuse.

---

## 🔬 Advanced Description

**Dataset:** 150 rows × 6 columns (`Id`, `SepalLengthCm`, `SepalWidthCm`, `PetalLengthCm`, `PetalWidthCm`, `Species`) — no missing values, no duplicates, perfectly balanced (50 samples per species).

**Exploratory Data Analysis (EDA):**
- Histogram + KDE plots for each feature (overall and split by species)
- Pairplot to visualize class separability across all feature combinations
- Per-species mean and standard deviation comparison for all 4 features

**Preprocessing:**
- `StandardScaler` applied to the 4 numeric features
- 80/20 train-test split (`random_state=42`)

**Models trained & compared:**

| Model | Configuration | Train Accuracy | Test Accuracy |
|---|---|---|---|
| KNN (baseline) | `n_neighbors=3` | 94.2% | 100% |
| SVM (baseline) | `kernel='rbf', C=1.0` | 97.5% | 100% |
| **SVM + GridSearchCV** ✅ (final) | tuned `C`, `kernel`, `gamma` (cv=5) | 96.7% | 96.7% |
| KNN + RandomizedSearchCV | tuned `n_neighbors`, `weights`, `p` (cv=5, 10 iters) | 100% | 100% |

**Why GridSearchCV-tuned SVM was chosen as the final model:** although the randomized-search KNN scored a perfect 100% on both train and test sets, a perfect score on a dataset this small (150 samples) is a strong overfitting signal rather than a sign of a better model. The GridSearchCV-tuned SVM's more moderate, consistent score was judged more trustworthy and generalizable.

**Model persistence:** the final `GridSearchCV` object (wrapping the tuned SVM) is saved with `joblib` as `Iris_DataSet.pkl`, alongside `Columns.pkl` storing the expected feature column order.

---

## 🧠 Model & Techniques Used

| Category | Technique |
|---|---|
| Data analysis | `pandas`, `df.describe()`, `value_counts()`, duplicate/null checks |
| Visualization | `seaborn` histplot, KDE, pairplot, `matplotlib` subplots |
| Feature scaling | `StandardScaler` |
| Data splitting | `train_test_split` |
| Classification models | `KNeighborsClassifier`, `SVC` (RBF & linear kernels) |
| Hyperparameter tuning | `GridSearchCV` (exhaustive), `RandomizedSearchCV` (randomized) |
| Evaluation | `accuracy_score`, train/test score comparison |
| Model serialization | `joblib` |

---

## 🚀 How to Use the Saved Model

```python
import joblib
import pandas as pd

# Load the trained model and expected feature order
model = joblib.load("Iris_DataSet.pkl")
columns = joblib.load("Columns.pkl")

# New sample: [SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm]
sample = pd.DataFrame([[5.1, 3.5, 1.4, 0.2]], columns=columns)

prediction = model.predict(sample)
print(prediction)  # e.g. ['Iris-setosa']
```

## 📦 Requirements

```
pandas
numpy
scikit-learn
seaborn
matplotlib
joblib
```

Install with:
```bash
pip install pandas numpy scikit-learn seaborn matplotlib joblib
```

---

## 📁 Project Structure

```
├── Iris_DataSet.ipynb   # Main notebook (EDA + training + tuning)
├── iris.csv             # Dataset
├── Iris_DataSet.pkl     # Final trained model (GridSearchCV + SVM)
├── Columns.pkl          # Feature column order
└── README.md
```

---

## 📄 License

Add your license here (e.g. MIT).
