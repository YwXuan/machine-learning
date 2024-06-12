from sklearn.model_selection import  train_test_split
from sklearn.svm import SVC
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import matplotlib
from distutils.version import LooseVersion
def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):

    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], 
                    y=X[y == cl, 1],
                    alpha=0.8, 
                    color=colors[idx],
                    marker=markers[idx], 
                    label=cl, 
                    edgecolor='black')

    # highlight test examples
    if test_idx:
        # plot all examples
        X_test, y_test = X[test_idx, :], y[test_idx]

        
        if LooseVersion(matplotlib.__version__) < LooseVersion('0.3.4'):
            plt.scatter(X_test[:, 0],
                        X_test[:, 1],
                        c='',
                        edgecolor='black',
                        alpha=1.0,
                        linewidth=1,
                        marker='o',
                        s=100, 
                        label='test set')
        else:
            plt.scatter(X_test[:, 0],
                        X_test[:, 1],
                        c='none',
                        edgecolor='black',
                        alpha=1.0,
                        linewidth=1,
                        marker='o',
                        s=100, 
                        label='test set')         
            

file_path = "best_buy_laptops_2024.csv"
data = pd.read_csv(file_path)
data = data.fillna(0)
data=data.drop(columns=['offers/priceCurrency'])
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
data=df.drop(columns=['features/1/description','features/0/description','model','brand'])


# 將顧客滿意度分為三類
df['satisfaction_level'] = pd.cut(df['aggregateRating/ratingValue'], bins=[-1, 3, 4, 5], labels=['low','medium','high'], right=False)

df.dropna(inplace=True)

mapping = {    'low': 1,
    'medium': 2,
    'high': 3
}
data = df[['brand_encoded','model_encoded','satisfaction_level']].replace(mapping)

X = data[['brand_encoded','model_encoded']]
y = data['satisfaction_level']

scaler = StandardScaler()
standard_x= scaler.fit_transform(X)

norm_data=pd.concat([X,y] , axis=1)
x_train, x_test, y_train, y_test=train_test_split( X ,y,test_size=0.2,random_state=42)
y_combined = np.hstack((y_train, y_test))

X_train_std = scaler.transform(x_train)
X_test_std = scaler.transform(x_test)
X_combined_std = np.vstack((X_train_std, X_test_std))


# 模型初始化
# svm = SVC(kernel='linear', C=1.0, random_state=1)
svm = SVC(kernel='rbf', random_state=1, gamma=0.10, C=10.0)
svm.fit(x_train, y_train)
plot_decision_regions(X_combined_std, 
                      y_combined,
                      classifier=svm, 
                      test_idx=range(105, 150))
plt.xlabel('brand [standardized]')
plt.ylabel('model  [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()
