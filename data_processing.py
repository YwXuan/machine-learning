import pandas as pd
from sklearn.preprocessing import LabelEncoder
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
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
  

def preprocess_data(file_path):
    # 讀取資料集
    data = pd.read_csv(file_path)
    data = data.fillna(0)
    data = data.drop(columns=['offers/priceCurrency'])
    df = pd.DataFrame(data)
    
    # 使用 LabelEncoder 將類別型資料轉換為數值型
    label_encoder = LabelEncoder()
    df['brand_encoded'] = label_encoder.fit_transform(df['brand'])
    df['model'] = df['model'].astype(str)
    df['model_encoded'] = label_encoder.fit_transform(df['model'])
    df['features/0/description'] = df['features/0/description'].astype(str)
    df['features/0/description_encoded'] = label_encoder.fit_transform(df['features/0/description'])
    df['features/1/description'] = df['features/1/description'].astype(str)
    df['features/1/description_encoded'] = label_encoder.fit_transform(df['features/1/description'])
    data = df.drop(columns=['features/1/description', 'features/0/description', 'model', 'brand'])
    
    # 將顧客滿意度分為三類
    df['satisfaction_level'] = pd.cut(df['aggregateRating/ratingValue'], bins=[-1, 3, 4, 5], labels=['low','medium','high'], right=False)

    df.dropna(inplace=True)

    mapping = {    'low': 1,
        'medium': 2,
        'high': 3
    }
    # data = df[['brand_encoded','model_encoded','satisfaction_level']].replace(mapping)
    data=pd.DataFrame(df, columns=['brand_encoded','model_encoded','satisfaction_level','aggregateRating/ratingValue','aggregateRating/reviewCount','offers/price','depth','features/0/description_encoded','features/1/description_encoded']).replace(mapping)

    return data

if __name__ == "__main__":
    file_path = "best_buy_laptops_2024.csv"
    processed_data = preprocess_data(file_path)
    print(processed_data.head())
