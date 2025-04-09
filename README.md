# Consumer Churn Prediction

This project focuses on improving an existing Random Forest model to predict customer churn more accurately, with a focus on boosting recall due to the cost of missing churned customers. Several key steps were taken to handle data imbalance, refine features, and compare model performance.

---

## 1. Categorical Variables

All categorical columns such as `Gender`, `Education_Level`, `Marital_Status`, etc., were label-encoded. One-hot encoding was avoided intentionally, as the tree-based models used in this project (Random Forest, XGBoost, LightGBM, and CatBoost) can handle label-encoded categories natively and do not require one-hot encoding.

---

## 2. Handling Class Imbalance

The target variable `Attrition_Flag` was highly imbalanced, with significantly fewer churned customers (value = 1). Two techniques were evaluated:

- Using `class_weight='balanced'` with Random Forest
- Applying **SMOTE (Synthetic Minority Oversampling Technique)** within each fold of cross-validation using an imblearn.pipeline.Pipeline to oversample the minority class before model training

Using 5-fold stratified cross-validation, Random Forest with SMOTE (applied within each fold to avoid data leakage) consistently achieved **higher recall** than the class-weighted approach. Since the business objective prioritizes minimizing false negatives (i.e., identifying all potential churns), SMOTE was selected for further modeling.

---

## 3. Feature Engineering

Based on visual pairplots and feature interaction insights, several **new features** were engineered to capture latent behavior patterns:

- `Trans_Power` = `Total_Trans_Ct` × `Total_Trans_Amt`  
- `Spending_Efficiency` = `Total_Trans_Amt` / `Total_Trans_Ct`  
- `Utilization_Gap` = `Credit_Limit` − `Avg_Open_To_Buy`  
- `Trans_Ct_per_Credit` = `Total_Trans_Ct` / `Credit_Limit`  
- `Trans_Amt_per_Change` = `Total_Trans_Amt` / `Total_Amt_Chng_Q4_Q1`  
- `Change_X_Util` = `Total_Ct_Chng_Q4_Q1` × `Avg_Utilization_Ratio`  
- `Stability_Index` = `Total_Ct_Chng_Q4_Q1` × `Total_Amt_Chng_Q4_Q1`  
- `Underutilized_Heavy_Spender` = 1 if `Avg_Utilization_Ratio` < 0.2 and `Total_Trans_Amt` > 10000 else 0  
- `Stable_Low_Use` = 1 if `Total_Ct_Chng_Q4_Q1` < 0.5 and `Total_Trans_Ct` < 30 else 0  
- `Log_Credit_Limit` = log-transformed `Credit_Limit` to reduce skewness

Removed least important features to prevent cluttering.

### 🧪 Manual Binning (Attempted and Dropped)

Initially, manual binning was explored on features like `Customer_Age` and `Avg_Open_To_Buy`, thinking it could improve interpretability or expose hidden buckets of behavior. However, it was observed that performance slightly dropped.

This is expected because **tree-based models already handle continuous variable splitting efficiently** and don’t benefit from manual discretization, which removes useful variance.

📌 **Conclusion**: Manual binning would reduce granularity and hinder the model’s ability to capture subtle decision boundaries.

---

## 4. Model Comparison (with SMOTE)

Using properly pipelined SMOTE within cross-validation, the following models were trained and evaluated using 5-fold stratified cross-validation:

- Random Forest
- XGBoost
- LightGBM
- CatBoost

Each model performed well. However:

- **CatBoost** achieved the highest accuracy and recall across folds, making it the top-performing model for churn prediction.

---

## 5. Results


- **All models performed very well** thanks to SMOTE improving minority class representation.
- **CatBoost** achieved the **highest accuracy** and **highest recall**, making it ideal for detecting churned customers.
- **LightGBM** achieved  the **Second highest recall**, making it ideal if the computing power is low.

In churn prediction tasks, **recall** is often more important than accuracy since **missing a churned customer can be costly**.

---

## 6. Conclusion

- We enhanced the performance compared to the original research paper, increasing accuracy from 95.71% to 96.50% and recall from 80.12% to 90.46%.

---

## 7. Future Work

- Perform **hyperparameter tuning** using GridSearchCV to optimize model performance.