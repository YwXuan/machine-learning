import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 读取数据
data = pd.read_csv('final2/1_descriptive_statistics/best_buy_laptops_2024.csv')
print("原始数据缺失值数量:")
print(data.isnull().sum())

# 處理缺失值
data = data.fillna(data.mean(numeric_only=True).round(2))

# 移除不需要的列
columns_to_drop = ['offers/priceCurrency', 'model', 'features/0/description', 'features/1/description']
data_processed = data.drop(columns=columns_to_drop)
print("處理後数据缺失值数量:")
print(data_processed.isnull().sum())

# 類別特徵轉換
brand_map = {'Acer': 1, 'Alienware': 2, 'ASUS': 3, 'Dell': 4, 'GIGABYTE': 5, 'HP': 6, 'HP OMEN': 7, 'Lenovo': 8, 'LG': 9, 'Microsoft': 10, 'MSI': 11, 'Razer': 12, 'Samsung': 13, 'Thomson': 14}
data_processed['brand'] = data_processed['brand'].map(brand_map)

# 特徵縮放
scaler = StandardScaler()
numeric_features = ['aggregateRating/reviewCount', 'offers/price', 'depth', 'width']
data_processed[numeric_features] = scaler.fit_transform(data_processed[numeric_features])

# 假设 'rating' 是你的目標變量
X = data_processed.drop(columns=['aggregateRating/ratingValue'])
y = data_processed['aggregateRating/ratingValue']

# 數據分割成訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=419)

# 定義模型函数
def grid_search_cv(model, param_grid, X_train, y_train):
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='r2')
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

# 初始化模型
linear_regression = LinearRegression()
ridge = Ridge()
random_forest = RandomForestRegressor(random_state=419)
xgboost = XGBRegressor(random_state=419)
svr = SVR()

# 调参和訓練模型
linear_regression.fit(X_train, y_train)  # 線性回歸不需要调参

ridge_param_grid = {'alpha': [0.1, 1, 10, 100]}
ridge = grid_search_cv(Ridge(), ridge_param_grid, X_train, y_train)

random_forest_param_grid = {'n_estimators': [100, 200], 'max_depth': [10, 20, 30]}
random_forest = grid_search_cv(RandomForestRegressor(random_state=419), random_forest_param_grid, X_train, y_train)

xgboost_param_grid = {'n_estimators': [100, 200], 'max_depth': [3, 6, 9], 'learning_rate': [0.01, 0.1, 0.3]}
xgboost = grid_search_cv(XGBRegressor(random_state=419), xgboost_param_grid, X_train, y_train)

svr_param_grid = {'C': [0.1, 1, 10, 100], 'gamma': ['scale', 'auto']}
svr = grid_search_cv(SVR(), svr_param_grid, X_train, y_train)

# 定義模型组合并訓練
models = {
    "Linear Regression": linear_regression,
    "Ridge": ridge,
    "RandomForest": random_forest,
    "XGBoost": xgboost,
    "SVR": svr
}

# 評估模型並繪製殘差圖
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(15, 15))
axes = axes.flatten()

for idx, (name, model) in enumerate(models.items()):
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    train_mse = mean_squared_error(y_train, train_pred)
    train_r2 = r2_score(y_train, train_pred)
    test_mse = mean_squared_error(y_test, test_pred)
    test_r2 = r2_score(y_test, test_pred)
    
    print(f"{name} - Train MSE: {train_mse}")
    print(f"{name} - Train R^2: {train_r2}")
    print(f"{name} - Test MSE: {test_mse}")
    print(f"{name} - Test R^2: {test_r2}")
    
    # 繪製殘差圖
    residuals = y_test - test_pred
    axes[idx].scatter(test_pred, residuals, label='Test Data')
    axes[idx].scatter(train_pred, y_train - train_pred, label='Train Data', alpha=0.5)
    axes[idx].hlines(y=0, xmin=min(test_pred), xmax=max(test_pred), colors='r', linestyles='--')
    axes[idx].set_title(f"{name} Residuals")
    axes[idx].set_xlabel("Predicted")
    axes[idx].set_ylabel("Residuals")
    axes[idx].legend()

# 移除多餘的子圖
if len(models) < len(axes):
    for j in range(len(models), len(axes)):
        fig.delaxes(axes[j])

plt.tight_layout(pad=5.0)  # 增加子圖之間的間距
plt.show()
