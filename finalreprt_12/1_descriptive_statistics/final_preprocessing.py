import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from collections import Counter

# 讀取數據
data = pd.read_csv('best_buy_laptops_2024.csv')
print(data.isnull().sum())

# 處理缺失值
data = data.fillna(round(data.mean(),2))

# 移除沒有用的類別特徵和其他不需要的列
data_processed = data.drop(columns=['offers/priceCurrency', 'model', 'features/0/description', 'features/1/description'])
print("處理後缺失值數量")
print(data_processed.isnull().sum())

# 類別特徵轉換
brand_map={'Acer':1,'Alienware':2,'ASUS':3,'Dell':4,'GIGABYTE':5,'HP':6,'HP OMEN':7,'Lenovo':8,'LG':9,'Microsoft':10,'MSI':11,'Razer':12,'Samsung':13,'Thomson':14}
data_processed['brand']=data_processed['brand'].map(brand_map)

# 等頻區間切分
data_processed['rating_bin'] = pd.qcut(data_processed['aggregateRating/ratingValue'], q=3, labels=['1', '2', '3'])

# 特徵縮放 
scaler = StandardScaler()
numeric_features = ['aggregateRating/reviewCount', 'offers/price', 'depth', 'width']
data_processed[numeric_features] = scaler.fit_transform(data_processed[numeric_features])

# 資料分割
X = data_processed.drop(columns=['aggregateRating/ratingValue', 'rating_bin'])
y = data_processed['rating_bin']

# SMOTE 
smote = SMOTE(k_neighbors=1,random_state=419)
X_resampled, y_resampled = smote.fit_resample(X, y)

print("原始數據分佈:", Counter(y))
print("SMOTE後數據分佈:", Counter(y_resampled))

# 數據分割成訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=419)

# 保存數據
data_resampled = pd.concat([pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled, name='aggregateRating/ratingValue')], axis=1)
data_resampled.to_csv('processed_dataset.csv', index=False)
