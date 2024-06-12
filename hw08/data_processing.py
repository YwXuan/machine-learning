import pandas as pd
from sklearn.preprocessing import LabelEncoder
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from numpy import interp
from distutils.version import LooseVersion
from sklearn.metrics import confusion_matrix,precision_score, recall_score, f1_score,roc_auc_score,auc,roc_curve
from sklearn.model_selection import StratifiedKFold


def Roc(model ,X_train,y_train,X_test,y_test,name):
    X_train = pd.DataFrame(X_train, columns=['brand_encoded','model_encoded','aggregateRating/ratingValue','aggregateRating/reviewCount',\
          'offers/price','depth','features/0/description_encoded','features/1/description_encoded'])
    X_test = pd.DataFrame(X_test, columns=['brand_encoded','model_encoded','aggregateRating/ratingValue','aggregateRating/reviewCount',\
          'offers/price','depth','features/0/description_encoded','features/1/description_encoded'])
    y_train = pd.DataFrame(y_train, columns=['satisfaction_level']) 
    y_test = pd.DataFrame(y_test, columns=['satisfaction_level']) 
    y_train = y_train.values.ravel()
    y_test = y_test.values.ravel()
    X_train2 = X_train.iloc[:, [0, 2]]
    cv = list(StratifiedKFold(n_splits=3).split(X_train, y_train))
    fig = plt.figure(figsize=(7, 5))
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    all_tpr = []
    for i, (train, test) in enumerate(cv):
        probas = model.fit(X_train2.iloc[train], y_train.iloc[train]).predict_proba(X_train2.iloc[test])
        fpr, tpr, thresholds = roc_curve(y_train.iloc[test], probas[:, 1], pos_label=1)
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label='ROC fold %d (area = %0.2f)' % (i+1, roc_auc))
    plt.plot([0, 1], [0, 1], linestyle='--', color=(0.6, 0.6, 0.6), label='Random guessing')
    mean_tpr /= len(cv)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, 'k--', label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)
    plt.plot([0, 0, 1], [0, 1, 1], linestyle=':', color='black', label='Perfect performance')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel(f'{name}-False positive rate')
    plt.ylabel(f'{name}-True positive rate')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()

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
  
def output(model, X_test, y_test):
    y_pred = model.predict_proba(X_test)
    y_pred_labels = np.argmax(y_pred, axis=1)
    cf = confusion_matrix(y_true=y_test, y_pred=y_pred_labels)
    acc = model.score(X_test, y_test)
    precision = precision_score(y_true=y_test, y_pred=y_pred_labels, average='weighted')
    recall = recall_score(y_true=y_test, y_pred=y_pred_labels, average='weighted')
    f1 = f1_score(y_true=y_test, y_pred=y_pred_labels , average='weighted')
    auc = roc_auc_score(y_test, y_pred,  multi_class='ovr')
    return cf, acc, precision, recall, f1, auc

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
