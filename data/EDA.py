import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("../house_prices.csv")

# Översikt
print(df.shape)
print(df.columns)
print(df.describe())
print(df.isnull().sum().sort_values(ascending=False).head(10))


corr = df.corr(numeric_only=True)
plt.figure(figsize=(12,10))
sns.heatmap(corr, cmap="coolwarm", annot=False)

sns.pairplot(df[["SalePrice", "GrLivArea", "TotalBsmtSF", "YearBuilt", "OverallQual"]])
