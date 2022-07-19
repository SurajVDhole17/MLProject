from pandas import *
from numpy import *
import joblib
from sklearn.linear_model import LinearRegression
ds = read_csv("dataset.csv")
x = ds[["YearsExperience"]]
y = ds[["Salary"]]
lm = LinearRegression()
lm.fit(x,y)
joblib.dump(lm,"salarymodel.pkl")
model = joblib.load("salarymodel.pkl")
p = model.predict([[5]])
print(p)