import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('./best_buy_laptops_2024.csv')
select_features=df[['aggregateRating/ratingValue','brand','depth','width']]
sns.pairplot(select_features, hue='brand', markers=["o", "s", "D"])

plt.show()
cols = df.select_dtypes(include=['int64', 'float64']).columns

# 計算相關係數矩陣
matrix = df[cols].corr()

# 繪製熱度圖
plt.figure(figsize=(10, 8))
sns.heatmap(matrix, annot=True, cmap='viridis', fmt=".2f", vmin=-1, vmax=1, alpha=0.7)
plt.title('correlation coefficient matrix')
plt.show()
