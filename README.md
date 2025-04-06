# Consumer Churn Prediction

This project focuses on improving an existing Random Forest model to predict customer churn more accurately, with a focus on boosting recall due to the cost of missing churned customers. Several key steps were taken to handle data imbalance, refine features, and compare model performance.

---

## 1. Categorical Variables

All categorical columns such as `Gender`, `Education_Level`, `Marital_Status`, etc., were label-encoded. One-hot encoding was avoided intentionally, as the tree-based models used in this project (Random Forest, XGBoost, LightGBM, and CatBoost) can handle label-encoded categories natively and do not require one-hot encoding.

---

## 2. Handling Class Imbalance

The target variable `Attrition_Flag` was highly imbalanced, with significantly fewer churned customers (value = 1). Two techniques were evaluated:

- Using `class_weight='balanced'` with Random Forest
- Applying **SMOTE (Synthetic Minority Oversampling Technique)** to oversample the minority class before model training

Using 5-fold cross-validation, Random Forest with SMOTE consistently achieved **higher recall** compared to using class weights. Since the business priority is to minimize false negatives (missed churns), SMOTE was chosen for further modeling.

---

## 3. Model Comparison (with SMOTE)

Using the SMOTE-resampled dataset, the following models were trained and evaluated using K-Fold cross-validation:

- Random Forest
- XGBoost
- LightGBM
- CatBoost

Each model performed well. However:

- **CatBoost** achieved the highest **accuracy**
- **Random Forest** achieved the highest **recall**

This makes Random Forest more favorable in churn scenarios where capturing all potential churned customers is critical.

---

## 4. Feature Importance & Binning

Feature importance was analyzed using Random Forest. Features like `Total_Trans_Amt`, `Avg_Utilization_Ratio`, and `Credit_Limit` ranked high.

While some features seemed like good candidates for binning (e.g., `Customer_Age`, `Avg_Open_To_Buy`), manual binning was **not used**. This is because tree-based models already perform **internal optimal splits** on continuous variables. Manual binning would reduce granularity and hinder the model’s ability to capture subtle decision boundaries.

---

## 5. Results


- **All models performed very well** thanks to SMOTE improving minority class representation.
- **CatBoost** achieved the **highest accuracy**.
- **Random Forest** achieved the **highest recall**, making it ideal for detecting churned customers.

In churn prediction tasks, **recall** is often more important than accuracy since **missing a churned customer can be costly**.
Random Forest slightly outperforms others in recall, making it the best option if the business goal is to detect as many churned customers as possible. CatBoost provides the best balance of accuracy and recall.

---

## 6. Recommendation

Use **Random Forest with SMOTE** when recall is the priority. This ensures that the model captures more of the actual churn cases, reducing the risk of losing customers who might have otherwise been retained.

---

## 7. Future Work

- Perform **hyperparameter tuning** using GridSearchCV or RandomizedSearchCV to optimize model performance.
- Visualize and analyze **confusion matrix heatmaps** for both CatBoost and Random Forest to better understand model predictions.
- Save the best-performing models using `joblib` or `pickle` so they can be reused directly without retraining:
  
  ```python
  import joblib
  joblib.dump(model, 'random_forest_smote.pkl')
  joblib.dump(catboost_model, 'catboost_smote.pkl')
