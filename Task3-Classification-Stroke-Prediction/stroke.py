import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


stroke_data = pd.read_csv('./healthcare-dataset-stroke-data.csv')

# Remove rows with NaN values in them
stroke_data.dropna(axis=0, inplace=True)
# Replace string values with 0 or 1
stroke_data['gender'].replace('Male', 0, inplace=True)
stroke_data['gender'].replace('Female', 1, inplace=True)
stroke_data['gender'].replace('Other', 1, inplace=True)

stroke_data["ever_married"].replace("Yes", 1, inplace=True)
stroke_data["ever_married"].replace("No", 0, inplace=True)

stroke_data["work_type"].replace("Private", 0, inplace=True)
stroke_data["work_type"].replace("Self-employed", 1, inplace=True)
stroke_data["work_type"].replace("Govt_job", 2, inplace=True)
stroke_data["work_type"].replace("children", 3, inplace=True)
stroke_data["work_type"].replace("Never_worked", 4, inplace=True)

stroke_data["Residence_type"].replace("Rural", 0, inplace=True)
stroke_data["Residence_type"].replace("Urban", 1, inplace=True)

stroke_data["smoking_status"].replace("formerly smoked", 0, inplace=True)
stroke_data["smoking_status"].replace("never smoked", 1, inplace=True)
stroke_data["smoking_status"].replace("smokes", 2, inplace=True)
stroke_data["smoking_status"].replace("Unknown", 3, inplace=True)

# Drop unnecessary id column
stroke_data.drop(['id'], axis=1, inplace=True)

X = stroke_data.drop('stroke', axis=1).values
y = stroke_data['stroke'].values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)


lr = LogisticRegression()
lr.fit(X_train, y_train)
predict = lr.predict(X_test)
print(classification_report(y_test, predict))
