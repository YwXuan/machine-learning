from xgboost import XGBClassifier
from data_processing import preprocess_data
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import numpy as np



data = pd.DataFrame(preprocess_data("best_buy_laptops_2024.csv"), \
        columns=['brand_encoded','model_encoded','satisfaction_level','aggregateRating/ratingValue'\
                ,'aggregateRating/reviewCount','offers/price','depth','features/0/description_encoded',\
                'features/1/description_encoded'])
X = data[['brand_encoded','model_encoded','aggregateRating/ratingValue','aggregateRating/reviewCount',\
        'offers/price','depth','features/0/description_encoded','features/1/description_encoded']]
y = data['satisfaction_level']


le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=1)


rf = RandomForestClassifier(criterion='gini',
                            n_estimators=25, 
                            random_state=1,
                            n_jobs=2)
rf.fit(X_train, y_train)
rf_feature_importances = rf.feature_importances_

# 建立XGBoost模型
xgb = XGBClassifier() 
xgb.fit(X_train, y_train)
xgb_feature_importances = xgb.feature_importances_


# 繪製特徵重要性
plt.figure(figsize=(10, 6))
plt.barh(np.arange(len(rf_feature_importances)), rf_feature_importances, tick_label=X.columns, color='b', alpha=0.7, label='Random Forest')
plt.barh(np.arange(len(xgb_feature_importances)), xgb_feature_importances, tick_label=X.columns, color='r', alpha=0.7, label='XGBoost')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance -- RF & XGBoost')
plt.tight_layout()
plt.legend()
plt.show()
