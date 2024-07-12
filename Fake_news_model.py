import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics

# Read file

df = pd.read_csv(r"C:\Users\Praval Patel\Desktop\vsproject\Ml supervise\FAKE NEW\news.csv")

# EDA

print(df.head(5))

print(df.tail(5))

print(df.info())

print(df.columns)

print(df.corr(numeric_only=True))

print(df.cov(numeric_only=True))

print(df.dtypes)

print(df.describe())

print(df.isnull().sum())

print(df.dropna(how='any', inplace = True))
print(df.isnull().sum())

#  using ladel encoder converting string to integer 

encoder = LabelEncoder()

df["title"] = encoder.fit_transform(df["title"])
df["text"] = encoder.fit_transform(df["text"])
df["label"] = encoder.fit_transform(df["label"])

print(df.dtypes)

df[(df >= 0).all(axis=1)]

# dividing two parts for prediction

x = df.drop(['label'],axis=1)
y = df["label"]

print(x.head(10))

print(y.head(10))

# splitting data train and test 

x_train, x_test, y_train, y_test = train_test_split(x , y, test_size=0.2, random_state=42)

print(x_train)

print(x_test)

print(y_train)

print(y_test)

# Fit Logistic Regreession model

zx = LogisticRegression ()

zx.fit(x_train,y_train)

tranning_data_prediction = zx.predict(x_train)

y_train = y_train[tranning_data_prediction >= 0]
tranning_data_prediction = tranning_data_prediction[tranning_data_prediction >= 0]

# counting model accuracy 

error_score = metrics.r2_score(y_train,tranning_data_prediction)*100
print("R squred error :", error_score)

# data visualisation

plt.scatter(y_train, tranning_data_prediction)
plt.grid()
plt.show()

# prediction test 

print(zx.predict([[8476,6155,1514]]))

print(zx.predict([[10294,5747,2185]]))

print(zx.predict([[3608,2946,5165]]))

print(zx.predict([[10142,653,5991]]))

print(zx.predict([[875,4788,2733]]))

print(zx.predict([[6903,4742,117]]))

print(zx.predict([[7341,2054,4168]]))
