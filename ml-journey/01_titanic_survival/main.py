import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt


def main():
    df = pd.read_csv("data/titanic.csv")
    
    df = df[["Survived", "Pclass", "Sex", "Age", "Fare"]].dropna()
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

    X = df.drop("Survived", axis=1)
    y = df["Survived"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=4)
    model = RandomForestClassifier(random_state=67)
    model.fit(X_train, y_train)


    pred = model.predict(X_test)
    print(classification_report(y_test, pred))
  

if __name__ == "__main__":
    main()
