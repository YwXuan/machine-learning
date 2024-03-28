import pandas as pd
from sklearn.preprocessing import LabelEncoder


# 作業內容
# 1. 處理missing data (刪除與填補資料) 
# 2. 每個特徵與Label的敘述統計 
# 3. 特徵縮放 
# 4. 類別特徵轉換 
# 5. 訓練與測試split data 

# 讀取CSV檔案

file_path = "best_buy_laptops_2024.csv"
data = pd.read_csv(file_path)

# # 檢視資料
# print(data.head())
# print(data.info())
# # 查看遺失數值
# print(data.isnull().sum())

# 將遺失值補齊
data = data.fillna(0)
print('1.確認數值是否有空值',data.isnull().sum())

# 檢視object欄位中的內容是否有意義
cols_obj = ['brand', 'model', 'offers/priceCurrency', 'features/0/description', 'features/1/description']

for col in cols_obj:
    print(f'Column {col}: {data[col].nunique()} sublevels')

# 輸出後發現offers/priceCurrency只表示貨幣種類(美元)
# 所以把這個欄位刪除
    
data=data.drop(columns=['offers/priceCurrency'])

# 查看整理後數值欄位的數據
print(data.describe())
new_cols_obj = ['brand', 'model', 'features/0/description', 'features/1/description']

# 查看類別obj欄位的數值種類有多少
for col in new_cols_obj:
    print(col,"：",data[col].nunique())

# 類別轉數值
df = pd.DataFrame(data)
label_encoder = LabelEncoder()
df['brand_encoded'] = label_encoder.fit_transform(df['brand'])

# 將 model 欄位的資料型態統一轉換成字符串
df['model'] = df['model'].astype(str)
# 使用標籤編碼(Label Encoding)將型號（model）轉換為數值
df['model_encoded'] = label_encoder.fit_transform(df['model'])

# 將 features/0/description 欄位的資料型態統一轉換成字符串
df['features/0/description'] = df['features/0/description'].astype(str)
# 使用標籤編碼(Label Encoding)將型號（features/0/description）轉換為數值
df['features/0/description_encoded'] = label_encoder.fit_transform(df['features/0/description'])

# 將 features/1/description 欄位的資料型態統一轉換成字符串
df['features/1/description'] = df['features/1/description'].astype(str)
# 使用標籤編碼(Label Encoding)將型號（features/1/description）轉換為數值
df['features/1/description_encoded'] = label_encoder.fit_transform(df['features/1/description'])
clean_data=df.drop(columns=['features/1/description','features/0/description','model','brand'])

print('4.類別轉數值',clean_data.head())

# 特徵縮放 敘述統計(次數、平均、中位、變異、標準差)
X = clean_data["brand_encoded"]
y = clean_data[["aggregateRating/ratingValue", "aggregateRating/reviewCount"]]
norm_data=pd.concat([X,y] , axis=1)

print('2.敘述統計',clean_data.describe())

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
standard_x=scaler.fit_transform(X.values.reshape(-1, 1))
standard_x_df=pd.DataFrame(standard_x, columns=['brand_encoded'])

print('3.特徵縮放：',standard_x_df)

# 區分資料 訓練/測試
from sklearn.model_selection import  train_test_split

x_train, x_test, y_train, y_test=train_test_split( X ,y,test_size=0.2,random_state=42)

print('5.切分後的比例：訓練：{%i}，測試：{%i}'%(len(x_train),len(x_test)))

