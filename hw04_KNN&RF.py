from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from data_processing import preprocess_data 
from data_processing import plot_decision_regions
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# 讀取並前處理資料
data = pd.DataFrame(preprocess_data("best_buy_laptops_2024.csv"), columns=['brand_encoded', 'model_encoded', 'satisfaction_level'])
X = data[['brand_encoded', 'model_encoded']]
y = data['satisfaction_level']

# 標準化特徵
scaler = StandardScaler()
X_std = scaler.fit_transform(X)
X_train_std, X_test_std, y_train, y_test = train_test_split(X_std, y, test_size=0.2, random_state=42)
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))


# 建立RF
forest = RandomForestClassifier(criterion='gini',
                                n_estimators=25, 
                                random_state=1,
                                n_jobs=2)
forest.fit(X_train_std, y_train)

# 繪製隨機森林的決策區域
plot_decision_regions(X_combined_std, y_combined, 
                      classifier=forest, test_idx=range(105, 150))

plt.xlabel('brand & model ')
plt.ylabel('satisfaction_level')
plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig('Rf.png', dpi=300)
plt.show()

# 建立K最近鄰分類器
knn = KNeighborsClassifier(n_neighbors=5, 
                           p=2, 
                           metric='minkowski')
knn.fit(X_train_std, y_train)

# 繪製K最近鄰的決策區域
plot_decision_regions(X_combined_std, y_combined, 
                      classifier=knn, test_idx=range(105, 150))

plt.xlabel('brand & model ')
plt.ylabel('satisfaction_level')
plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig('knn.png', dpi=300)
plt.show()
