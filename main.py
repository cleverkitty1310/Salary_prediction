import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LassoCV
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import mean_squared_error

sns.set(rc={'figure.figsize': [15, 7]}, font_scale=1.2)
sns.set_style('whitegrid')

import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("./ds_salaries.csv", index_col=[0])

experience = {
    "EN": 0,
    "MI": 1,
    "SE": 2,
    "EX": 3
}

df["experience_level"] = df["experience_level"].map(experience)
df["job_title"].value_counts()

def Titles_reduction(x):
    if x.find("Data Science") >= 0 or x.find("Data Scientist") >= 0:
        df["job_title"].replace(x, "Data Scientist", inplace=True)
    elif x.find("Analyst") >= 0 or x.find("Anayltics") >= 0:
        df["job_title"].replace(x, "Data Analyst", inplace=True)
    elif x.find("ML") >= 0 or x.find("Machine Learning") >= 0:
        df["job_title"].replace(x, "Machine Learning Engineer", inplace=True)
    elif x.find("Data Engineer") >= 0 or x.find("Data Engineering") >= 0:
        df["job_title"].replace(x, "Data Engineer", inplace=True)
    else:
        df["job_title"].replace(x, "AI related", inplace=True)

for i in df["job_title"]:
    Titles_reduction(i)

df["job_title"].value_counts()

size = {
    "S": 0,
    "M": 1,
    "L": 2
}

df["company_size"] = df["company_size"].map(size)

def res(x):
    if x =="US":
        return "US"
    else:
        return "Other"

df["employee_residence"] = df["employee_residence"].apply(res)

df["company_location"] = df["company_location"].apply(res)

df["salary_in_usd"] = np.log(df["salary_in_usd"])
df["salary_in_usd"]

nums = df.select_dtypes(exclude="object").columns

cats = df.select_dtypes(include="object").columns

df = pd.get_dummies(df, columns=cats, drop_first=True)

X, y = df.drop("salary", axis=1), df["salary"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

nums = nums.drop("salary")

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train[nums])

X_train[nums] = scaler.transform(X_train[nums])
X_test[nums] = scaler.transform(X_test[nums])

models = {
    "Linear regression": LinearRegression(),
    "Lasso": LassoCV(),
    "Ridge": RidgeCV(),
    "ElasticNet": ElasticNetCV()
}


Results = {
    "Model": [],
    "Train Score": [],
    "Test Score": [],
    "RMSE": []
}


for name, model in models.items():
    model.fit(X_train, np.log(y_train))
    train_s = model.score(X_train, np.log(y_train))
    test_s = model.score(X_test, np.log(y_test))
    y_pred = model.predict(X_test)
    RMSE = mean_squared_error((y_pred), np.log(y_test))
    Results["Model"].append(name)
    Results["Train Score"].append(train_s)
    Results["Test Score"].append(test_s)
    Results["RMSE"].append(RMSE)
    print("Model: ", name)
    print("Train Score: ", train_s)
    print("Test Score: ", test_s)
    print("RMSE: ", round(RMSE, 2))
    print("==================================")


scores = pd.DataFrame(Results)

tidy = scores.melt(id_vars="Model").rename(columns=str.title)

sns.barplot(data=tidy, x="Variable", y="Value", hue="Model")
plt.show()