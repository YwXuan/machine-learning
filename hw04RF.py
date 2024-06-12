from sklearn.ensemble import RandomForestClassifier
from data_processing import preprocess_data 
from data_processing import plot_decision_regions
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import  train_test_split
import pandas as pd
import numpy as np

# 匯入模型進行資料前處理
data = pd.DataFrame(preprocess_data("best_buy_laptops_2024.csv"),columns=['brand_encoded','model_encoded','satisfaction_level'])

# 將資料區分
X = data[['brand_encoded','model_encoded']]
y = data['satisfaction_level']

# 進行標準化
scaler = StandardScaler()
standard_x= scaler.fit_transform(X)

norm_data=pd.concat([X,y] , axis=1)
x_train, x_test, y_train, y_test=train_test_split( X ,y,test_size=0.2,random_state=42)
y_combined = np.hstack((y_train, y_test))
X_combined = np.vstack((x_train, x_test))


forest = RandomForestClassifier(criterion='gini',
                                n_estimators=25, 
                                random_state=1,
                                n_jobs=2)
forest.fit(x_train, y_train)

plot_decision_regions(X_combined, y_combined, 
                      classifier=forest, test_idx=range(105, 150))

plt.xlabel('brand & model')
plt.ylabel('satisfaction_level')
plt.legend(loc='upper left')
plt.tight_layout()
#plt.savefig('images/03_22.png', dpi=300)
plt.show()