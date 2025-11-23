import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle

df = pd.read_csv("student_data.csv")
X = df.drop("Final Score", axis=1)
y = df["Final Score"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor()
model.fit(X_train, y_train)

pickle.dump(model, open("student_model.pkl", "wb"))
print("Model trained and saved!")
