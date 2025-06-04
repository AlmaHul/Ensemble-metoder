import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(path="house_prices.csv"):
    df = pd.read_csv(path)
    df = df.select_dtypes(include=["number"]).dropna()
    X = df.drop("SalePrice", axis=1)
    y = df["SalePrice"]
    return train_test_split(X, y, test_size=0.2, random_state=42)
