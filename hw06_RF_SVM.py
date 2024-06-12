from data_processing import preprocess_data
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, f1_score

from sklearn.metrics import roc_curve, auc
from numpy import interp
from sklearn.model_selection import StratifiedKFold


data = pd.DataFrame(preprocess_data("best_buy_laptops_2024.csv"), \
        columns=['brand_encoded','model_encoded','satisfaction_level','aggregateRating/ratingValue'\
                ,'aggregateRating/reviewCount','offers/price','depth','features/0/description_encoded',\
                'features/1/description_encoded'])
X = data[['brand_encoded','model_encoded','aggregateRating/ratingValue','aggregateRating/reviewCount',\
          'offers/price','depth','features/0/description_encoded','features/1/description_encoded']]
y = data['satisfaction_level']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=1)

forest = RandomForestClassifier(criterion='gini',
                                n_estimators=25, 
                                random_state=1,
                                n_jobs=2)
forest.fit(X_train, y_train)
feat_labels = data.columns[1:]

importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]
# 輸出所有特徵值的重要性指數
for f in range(X_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, 
                            feat_labels[indices[f]], 
                            importances[indices[f]]))

import_5 = []
# 選擇符合重要性指數條件(前三名)的並用迴圈分別輸出
sfm = SelectFromModel(forest, threshold=-np.inf,max_features=3 , prefit=True)
X_selected = sfm.transform(X_train)
for f in range(X_selected.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, 
                            feat_labels[indices[f]], 
                            importances[indices[f]]))
    import_5.append(feat_labels[indices[f]])
    
# 將前三名特徵放入模型預測
X = data[import_5]
y = data['satisfaction_level']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=1)

# SVM模型

svm = make_pipeline(StandardScaler(),
                    SVC(random_state=1,probability=True))

svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)
confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)

fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')

plt.xlabel('SVM-Predicted label')
plt.ylabel('SVM-True label')

plt.tight_layout()
plt.show()

print('SVM-Test Accuracy: %.3f' % svm.score(X_test, y_test))
print('SVM-Precision: %.3f' % precision_score(y_true=y_test, y_pred=y_pred,average='weighted', zero_division=1))
print('SVM-Recall: %.3f' % recall_score(y_true=y_test, y_pred=y_pred ,average='weighted'))
print('SVM-F1: %.3f' % f1_score(y_true=y_test, y_pred=y_pred ,average='weighted'))


X_train2 = X_train.iloc[:, [0, 2]]
cv = list(StratifiedKFold(n_splits=3).split(X_train, y_train))
kfold = StratifiedKFold(n_splits=10).split(X_train, y_train)

fig = plt.figure(figsize=(7, 5))
mean_tpr = 0.0
mean_fpr = np.linspace(0, 1, 100)
all_tpr = []

# 遍歷每個交叉驗證折疊
for i, (train, test) in enumerate(cv):
    # 訓練模型並獲取預測概率
    probas = svm.fit(X_train2.iloc[train],
                         y_train.iloc[train]).predict_proba(X_train2.iloc[test])
    
    # 計算 ROC 曲線的指標
    fpr, tpr, thresholds = roc_curve(y_train.iloc[test],
                                     probas[:, 1],
                                     pos_label=1)
    mean_tpr += interp(mean_fpr, fpr, tpr)
    mean_tpr[0] = 0.0
    roc_auc = auc(fpr, tpr)

    # 繪製 ROC 曲線
    plt.plot(fpr,
             tpr,
             label='ROC fold %d (area = %0.2f)'
                   % (i+1, roc_auc))

plt.plot([0, 1],
         [0, 1],
         linestyle='--',
         color=(0.6, 0.6, 0.6),
         label='Random guessing')

mean_tpr /= len(cv)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
plt.plot(mean_fpr, mean_tpr, 'k--',
         label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)
plt.plot([0, 0, 1],
         [0, 1, 1],
         linestyle=':',
         color='black',
         label='Perfect performance')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('svm-False positive rate')
plt.ylabel('svm-True positive rate')
plt.legend(loc="lower right")

plt.tight_layout()
plt.show()

# k = 5 交叉驗證

kfold = StratifiedKFold(n_splits=5).split(X_train, y_train)

scores = []
for k, (train, test) in enumerate(kfold):
    svm.fit(X_train.iloc[train], y_train.iloc[train])
    score = svm.score(X_train.iloc[test], y_train.iloc[test])
    scores.append(score)
    print('Fold: %2d, Class dist.: %s, Acc: %.3f' % (k+1,
          np.bincount(y_train.iloc[train]), score))
    
print('\nCV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))