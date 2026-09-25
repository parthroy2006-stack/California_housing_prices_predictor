# California_housing_prices_predictor

An end-to-end machine learning project for predicting California housing prices using regression models and the California Housing Prices dataset sourced from Kaggle.

## 📌 Project Overview

The goal of this project is to build a machine learning pipeline capable of predicting **median house values** based on various housing and demographic features.

The project includes data preprocessing, stratified train-test splitting, model comparison, cross-validation, model persistence, and an inference pipeline.

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Joblib

## 📊 Dataset

The project uses the **California Housing Prices** dataset sourced from Kaggle.

The target variable is:

* `median_house_value`

The dataset contains numerical housing/demographic features along with the categorical feature:

* `ocean_proximity`

## ⚙️ Machine Learning Workflow

### 1. Data Splitting

A stratified train-test split is created based on `median_income` categories to maintain a similar income distribution across the training and test sets.

### 2. Data Preprocessing

The preprocessing pipeline includes:

* Median imputation for missing numerical values
* Standardization using `StandardScaler`
* One-hot encoding of categorical features
* `ColumnTransformer` to combine numerical and categorical preprocessing

### 3. Model Comparison

Three regression algorithms were trained and evaluated:

* Linear Regression
* Decision Tree Regressor
* Random Forest Regressor

Models were evaluated using **10-fold cross-validation** with **Root Mean Squared Error (RMSE)**.

### 4. Final Model

The Random Forest Regressor was used as the final model and trained on the prepared training data.

The trained model and preprocessing pipeline are saved using **Joblib**, allowing them to be reused for predictions without retraining.

### 5. Inference

The inference workflow:

1. Loads the saved model and preprocessing pipeline.
2. Reads new housing data.
3. Applies the same preprocessing steps.
4. Generates housing price predictions.
5. Saves the predictions to `predictions.csv`.

## 📁 Project Structure

```text
California-Housing-Price-Prediction/
│
├── housing.csv
├── input.csv
├── predictions.csv
├── model.pkl
├── pipeline.pkl
├── main.py
├── train.py
└── README.md
```

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd California-Housing-Price-Prediction
```

### 2. Install dependencies

```bash
pip install pandas numpy scikit-learn joblib
```

### 3. Train the model

Run the training script:

```bash
python train.py
```

This creates:

```text
model.pkl
pipeline.pkl
```

### 4. Generate predictions

Provide new housing data in `input.csv` and run:

```bash
python train.py
```

The predictions will be saved to:

```text
predictions.csv
```

## 📈 Evaluation

The models were compared using **10-fold cross-validation and RMSE**. Lower RMSE indicates better predictive performance.

## 🔮 Future Improvements

* Perform systematic hyperparameter tuning using GridSearchCV or RandomizedSearchCV.
* Add additional regression models for comparison.
* Analyze feature importance from the Random Forest model.
* Deploy the model using Flask or Streamlit.
* Add visualizations for exploratory data analysis and model performance.

## 👨‍💻 Author

**Parth Roy**


