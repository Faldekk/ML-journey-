import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

DataFrame = pd.read_csv("titanic.csv")
DataFrame = DataFrame[["Survived", "Pclass", "Sex", "Age", "Fare"]].dropna()
DataFrame["Sex"] = DataFrame["Sex"].map({"male": 0, "female": 1})

X = DataFrame.drop("Survived", axis=1)
y = DataFrame["Survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=41
)

model = RandomForestClassifier(random_state=41)
model.fit(X_train, y_train)

pred = model.predict(X_test)
print(classification_report(y_test, pred))
