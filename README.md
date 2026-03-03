Loan Approval Prediction using Machine Learning

A Machine Learning project that predicts whether a loan application will be Approved (Y) or Rejected (N) based on applicant details.

This project demonstrates:

End-to-end ML pipeline building

Data preprocessing

Feature engineering

Model training & evaluation

Hyperparameter tuning

Model saving for deployment

📌 Problem Statement

Financial institutions need to automate loan approval decisions based on customer information such as income, education, credit history, and loan amount.

The goal is to build a classification model that predicts:

Loan_Status:
Y → Approved
N → Rejected

📂 Dataset Features

Typical features used:

Gender

Married

Dependents

Education

Self_Employed

ApplicantIncome

CoapplicantIncome

LoanAmount

Loan_Amount_Term

Credit_History

Property_Area

Target variable:

Loan_Status

🛠️ Tech Stack

Python 🐍

Pandas

NumPy

Scikit-learn

Matplotlib / Seaborn

Joblib

⚙️ Machine Learning Pipeline

The project uses Scikit-learn Pipeline & ColumnTransformer for clean preprocessing.

🔹 Preprocessing

Numeric Features

Median Imputation

Standard Scaling

Categorical Features

Most Frequent Imputation

Ordinal Encoding (handles unknown values)

🤖 Models Used

Logistic Regression

Random Forest

Gradient Boosting

(Optional) XGBoost

Hyperparameter tuning done using:

GridSearchCV

📊 Model Evaluation

Evaluation Metrics:

Accuracy

Precision

Recall

F1-Score

ROC-AUC Score

Confusion Matrix
How to Run the Project

Clone the repository

git clone https://github.com/your-username/loan-prediction.git


Install dependencies

pip install -r requirements.txt


Run the training script

python train.py
