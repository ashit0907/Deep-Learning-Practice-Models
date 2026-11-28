# Credit Card Customer Churn Prediction (Keras ANN)

Practice binary classification project using a TensorFlow/Keras Artificial Neural Network to predict whether a bank customer exits (`Exited`) based on demographic, account, and activity features.

## Dataset & Model Specs
- **Dataset:** `Churn_Modelling.csv` via `kagglehub` (`rjmanoj/credit-card-customer-churn-prediction`, excluded from repository)
- **Raw Shape:** `10,000` rows × `14` columns (`1.1+ MB` in memory)
- **Data Quality:** `0` missing values across all columns; `0` duplicate rows
- **Preprocessing:** Dropped `RowNumber`, `CustomerId`, and `Surname`; one-hot encoded `Geography` and `Gender` (`drop_first=True`); `80/20` stratified train-test split (`random_state=42`); `StandardScaler`
- **Architecture:** `Input(11)` → `Dense(3, activation="sigmoid")` (`36` params) → `Dense(1, activation="sigmoid")` (`4` params)
- **Total Trainable Parameters:** `40` (`160.00 B`; `0` non-trainable)
- **Training & Evaluation:** `binary_crossentropy` loss, `adam` optimizer, `10` epochs (`Epoch 10/10` loss: `0.4302`); threshold `> 0.5` achieved **Test Accuracy: `0.8025`**

## How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Run `credit_card_churn_ann.ipynb` (downloads `Churn_Modelling.csv` via `kagglehub`; trains dynamically with no pre-saved model file).