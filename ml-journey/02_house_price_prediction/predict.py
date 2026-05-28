import pandas as pd
import joblib

DATA_PATH = "data/housing.csv"
MODEL_PATH = "housing_model.pkl"

def main():
    try:
        model = joblib.load(MODEL_PATH)
    except FileNotFoundError:
        print(f"Error: {MODEL_PATH} not found. Run train.py first.")
        return
    
    df = pd.read_csv(DATA_PATH, header=None)
    
    example_row = df.iloc[[3], :-1]
    actual_value = df.iloc[0, -1]

    prediction = model.predict(example_row)

    print(f"Example features:\n{example_row}")
    print(f"Predicted value: {prediction[0]}")
    print(f"Actual value: {actual_value}")

if __name__ == "__main__":
    main()
