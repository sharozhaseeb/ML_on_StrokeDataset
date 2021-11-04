import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB


df = pd.read_csv("../input/stroke-prediction-dataset/healthcare-dataset-stroke-data.csv")

df.head(10)

df.shape

df.info()

cols_to_be_object = ["hypertension","heart_disease"]
for col in cols_to_be_object:
    df[col] = df[col].astype("object")


df.describe().T

df.isnull().sum()
df["bmi"].fillna(df["bmi"].median(), inplace=True)
df.duplicated().sum()
for col in df.columns:
    print(df[col].value_counts())
    print("-"*15)

df["smoking_status"].replace("Unknown", df["smoking_status"].mode().values[0], inplace=True)

print(df["smoking_status"].value_counts())

plt.figure(figsize=(10,6))
sns.countplot(df["gender"])
plt.title("Gender", size=15)
plt.show()

plt.figure(figsize=(10,6))
sns.distplot(df["age"])
plt.title("Age", size=15)
plt.show()

plt.figure(figsize=(10,6))
sns.countplot(df["heart_disease"])
plt.title("Heart Disease Numbers", size=15)
plt.show()


plt.figure(figsize=(10,6))
sns.barplot(x=df["stroke"], y=df["bmi"])
plt.title("BMI vs Stroke", size=15)
plt.show()

plt.figure(figsize=(10,6))
sns.barplot(x=df["smoking_status"], y=df["stroke"])
plt.title("Smoking Status vs Stroke", size=15)
plt.show()

plt.figure(figsize=(10,6))
sns.barplot(x=df["stroke"], y=df["hypertension"])
plt.title("Hypertension vs Stroke", size=15)
plt.show()

X = df.drop(["id","stroke"], axis=1)
y = df["stroke"]

y = pd.DataFrame(y, columns=["stroke"])

display(X.head())
display(y.head())

numerical_cols = X.select_dtypes(["float64","int64"])
scaler = StandardScaler()
X[numerical_cols.columns] = scaler.fit_transform(X[numerical_cols.columns])
X.head()

categorical_cols = X.select_dtypes("object")
X = pd.get_dummies(X, columns=categorical_cols.columns)

X.head()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

models = pd.DataFrame(columns=["Model","Accuracy Score"])

log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
predictions = log_reg.predict(X_test)
score = accuracy_score(predictions, y_test)
print(f"LogisticRegression: {score}")

new_row={"Model": "LogisticRegression", "Accuracy Score": score}
models = models.append(new_row, ignore_index=True)

GNB = GaussianNB()
GNB.fit(X_train, y_train)
predictions = GNB.predict(X_test)
score = accuracy_score(predictions, y_test)
print(f"GaussianNB: {score}")

new_row={"Model": "GaussianNB", "Accuracy Score": score}
models = models.append(new_row, ignore_index=True)

BNB = BernoulliNB()
BNB.fit(X_train, y_train)
predictions = BNB.predict(X_test)
score = accuracy_score(predictions, y_test)
print(f"BernoulliNB: {score}")

new_row={"Model": "BernoulliNB", "Accuracy Score": score}
models = models.append(new_row, ignore_index=True)

svm = SVC()
svm.fit(X_train, y_train)
predictions = svm.predict(X_test)
score = accuracy_score(predictions, y_test)
print(f"SVC: {score}")

new_row={"Model": "SVC", "Accuracy Score": score}
models = models.append(new_row, ignore_index=True)

randomforest = RandomForestClassifier(n_estimators=1000, random_state=42)
randomforest.fit(X_train, y_train)
predictions = randomforest.predict(X_test)
score = accuracy_score(predictions, y_test)
print(f"RandomForestClassifier: {score}")

new_row={"Model": "RandomForestClassifier", "Accuracy Score": score}
models = models.append(new_row, ignore_index=True)

xgb = XGBClassifier(use_label_encoder=False)
xgb.fit(X_train, y_train)
predictions = xgb.predict(X_test)
score = accuracy_score(predictions, y_test)
print(f"XGBClassifier: {score}")

new_row={"Model": "XGBClassifier", "Accuracy Score": score}
models = models.append(new_row, ignore_index=True)

models.sort_values(by="Accuracy Score", ascending=False)




































