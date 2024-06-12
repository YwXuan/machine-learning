import pandas as pd

# 讀取 CSV 檔案
file_path = './final/best_buy_laptops_2024.csv'
df = pd.read_csv(file_path)

# 顯示前五行資料
print("前五行資料:")
print(df.head())

# 資料概況
print("\n資料概況:")
print(df.info())

# 統計摘要
print("\n統計摘要:")
print(df.describe())

# 缺失值統計
print("\n缺失值統計:")
print(df.isnull().sum())

# 計算各類別的分佈 (假設有一個名為 'category' 的欄位)
if 'category' in df.columns:
    print("\n各類別的分佈:")
    print(df['category'].value_counts())

# 計算數值型欄位的平均值、中位數、標準差
numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns

for col in numerical_columns:
    print(f"\n欄位 {col} 的統計數據:")
    print(f"平均值: {df[col].mean()}")
    print(f"中位數: {df[col].median()}")
    print(f"標準差: {df[col].std()}")

# 顯示數值型欄位的分佈圖
import matplotlib.pyplot as plt
import seaborn as sns

for col in numerical_columns:
    plt.figure(figsize=(10, 5))
    sns.histplot(df[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.show()