import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 讀取CSV資料
file_path = 'processed_dataset.csv'
data = pd.read_csv(file_path)

# 指定特徵和標籤列
features = ['brand', 'aggregateRating/reviewCount', 'offers/price', 'depth', 'width']
target = 'aggregateRating/ratingValue'

# 分離特徵和標籤
X = data[features]
y = data[target]

# 分割資料集為訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 類別特徵與數值特徵分離
categorical_features = ['brand']
numerical_features = ['aggregateRating/reviewCount', 'offers/price', 'depth', 'width']

# 構建預處理管道，增加多項式特徵
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('scaler', StandardScaler()),
            ('poly', PolynomialFeatures(degree=2, include_bias=False))
        ]), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# 定義模型
models = {
    'Linear Regression': LinearRegression(),
    'Lasso': Lasso(),
    'Ridge': Ridge(),
    'ElasticNet': ElasticNet(),
    'SVR': SVR(),
    'Random Forest': RandomForestRegressor(),
    'Gradient Boosting': GradientBoostingRegressor(),
    'XGBoost': xgb.XGBRegressor(),
    'CatBoost': CatBoostRegressor(verbose=0)
}

# 訓練和評估模型
results = {}
residuals = {}
short_names = {
    'Linear Regression': 'LR',
    'Lasso': 'Lasso',
    'Ridge': 'Ridge',
    'ElasticNet': 'ElasticNet',
    'SVR': 'SVR',
    'Random Forest': 'RF',
    'Gradient Boosting': 'GB',
    'XGBoost': 'XGB',
    'CatBoost': 'CatB'
}

for name, model in models.items():
    # 構建管道
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
    
    # 網格搜索超參數調整（僅適用於部分模型）
    if name in ['Lasso', 'Ridge', 'ElasticNet', 'SVR', 'Random Forest', 'Gradient Boosting', 'XGBoost', 'CatBoost']:
        params = {
            'Lasso': {'model__alpha': [0.01, 0.1, 1, 10, 100]},
            'Ridge': {'model__alpha': [0.01, 0.1, 1, 10, 100]},
            'ElasticNet': {'model__alpha': [0.01, 0.1, 1, 10, 100], 'model__l1_ratio': [0.2, 0.5, 0.8]},
            'SVR': {'model__C': [0.1, 1, 10, 100], 'model__gamma': ['scale', 'auto']},
            'Random Forest': {'model__n_estimators': [100, 200, 300], 'model__max_depth': [10, 20, 30]},
            'Gradient Boosting': {'model__n_estimators': [100, 200, 300], 'model__learning_rate': [0.01, 0.1, 0.2]},
            'XGBoost': {'model__n_estimators': [100, 200, 300], 'model__learning_rate': [0.01, 0.1, 0.2], 'model__max_depth': [3, 6, 9]},
            'LightGBM': {'model__n_estimators': [100, 200, 300], 'model__learning_rate': [0.01, 0.1, 0.2], 'model__num_leaves': [31, 62, 127], 'model__max_depth': [-1, 10, 20]},
            'CatBoost': {'model__iterations': [100, 200, 300], 'model__learning_rate': [0.01, 0.1, 0.2], 'model__depth': [3, 6, 9]}
        }
        grid = GridSearchCV(pipeline, params[name], cv=5, n_jobs=-1)
        grid.fit(X_train, y_train)
        pipeline = grid.best_estimator_
    else:
        pipeline.fit(X_train, y_train)
    
    # 預測
    y_pred = pipeline.predict(X_test)
    
    # 確保預測結果為一維數組
    if y_pred.ndim > 1:
        y_pred = y_pred.ravel()
    
    # 計算殘差
    residuals[short_names[name]] = y_test - y_pred
    
    # 評估指標
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    results[short_names[name]] = {'MSE': mse, 'R^2': r2}
    print(f"{name} - MSE: {mse}, R^2: {r2}")

# 結果可視化
models_names = list(results.keys())
mse_values = [results[name]['MSE'] for name in models_names]
r2_values = [results[name]['R^2'] for name in models_names]

plt.figure(figsize=(14, 6))

# MSE圖
plt.subplot(1, 2, 1)
plt.bar(models_names, mse_values, color='skyblue')
plt.title('Mean Squared Error (MSE)')
plt.ylabel('MSE')
plt.xlabel('Model')

# R^2圖
plt.subplot(1, 2, 2)
plt.bar(models_names, r2_values, color='lightgreen')
plt.title('R^2 Score')
plt.ylabel('R^2')
plt.xlabel('Model')

plt.tight_layout()
plt.show()

# 殘差圖 - 第一頁
plt.figure(figsize=(15, 10))
for i, (name, residual) in enumerate(list(residuals.items())[:6], 1):
    plt.subplot(3, 2, i)
    plt.scatter(y_test, residual)
    plt.axhline(0, color='red', linestyle='--', lw=2)
    plt.title(f'{name} Residuals')
    plt.xlabel('Actual Values')
    plt.ylabel('Residuals')

plt.tight_layout()
plt.show()

# 殘差圖 - 第二頁
plt.figure(figsize=(15, 10))
for i, (name, residual) in enumerate(list(residuals.items())[6:], 1):
    plt.subplot(3, 2, i)
    plt.scatter(y_test, residual)
    plt.axhline(0, color='red', linestyle='--', lw=2)
    plt.title(f'{name} Residuals')
    plt.xlabel('Actual Values')
    plt.ylabel('Residuals')

plt.tight_layout()
plt.show()

plt.figure(figsize=(15, 7))
plt.subplot(1, 2, 1)
plt.scatter(range(len(y_train)), y_train, color='blue', alpha=0.5, label='Training data')
plt.scatter(range(len(y_train)), pipeline.predict(X_train), color='red', alpha=0.5, label='Training prediction')
plt.xlabel('Index')
plt.ylabel('Rating Value')
plt.title('Training Data vs. Prediction')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(range(len(y_test)), y_test, color='blue', alpha=0.5, label='Test data')
plt.scatter(range(len(y_test)), y_pred, color='red', alpha=0.5, label='Test prediction')
plt.xlabel('Index')
plt.ylabel('Rating Value')
plt.title('Test Data vs. Prediction')
plt.legend()

plt.tight_layout()
plt.show()
