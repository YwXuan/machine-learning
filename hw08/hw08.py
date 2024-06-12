# 2. RF , Adaboost 及 XGBoost 建立模型, 混淆矩陣, Accuracy, Precision, Recall, F1, AUC, ROC 繳交程式與Excel
from data_processing import preprocess_data ,output
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import AdaBoostClassifier ,RandomForestClassifier
from numpy import interp
import matplotlib.pyplot as plt
from sklearn.metrics import auc,roc_curve
from sklearn.model_selection import StratifiedKFold
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

rf_cf, rf_acc, rf_precision, rf_recall, rf_f1, rf_auc = output(rf,X_test, y_test)
print("Random Forest:")
print("Confusion Matrix:")
print(rf_cf)
print("Accuracy:", rf_acc)
print("Precision:", rf_precision)
print("Recall:", rf_recall)
print("F1 Score:", rf_f1)
print("AUC:", rf_auc)

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
    probas = rf.fit(X_train2.iloc[train],
                         y_train[train]).predict_proba(X_train2.iloc[test])
    
    # 計算 ROC 曲線的指標
    fpr, tpr, thresholds = roc_curve(y_train[test],
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
plt.xlabel('Random Forest:False positive rate')
plt.ylabel('Random Forest:True positive rate')
plt.legend(loc="lower right")

plt.tight_layout()
plt.show()

# 建立XGBoost模型
xgb = XGBClassifier() 
xgb.fit(X_train, y_train)

xgb_cf, xgb_acc, xgb_precision, xgb_recall, xgb_f1, xgb_auc = output(xgb,X_test, y_test)
print("xgboost:")
print("Confusion Matrix:")
print(xgb_cf)
print("Accuracy:", xgb_acc)
print("Precision:", xgb_precision)
print("Recall:", xgb_recall)
print("F1 Score:", xgb_f1)
print("AUC:", xgb_auc)

fig = plt.figure(figsize=(7, 5))
mean_tpr = 0.0
mean_fpr = np.linspace(0, 1, 100)
all_tpr = []

# 遍歷每個交叉驗證折疊
for i, (train, test) in enumerate(cv):
    # 訓練模型並獲取預測概率
    probas = rf.fit(X_train2.iloc[train],
                         y_train[train]).predict_proba(X_train2.iloc[test])
    
    # 計算 ROC 曲線的指標
    fpr, tpr, thresholds = roc_curve(y_train[test],
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
plt.xlabel('xgboost :False positive rate')
plt.ylabel('xgboost :True positive rate')
plt.legend(loc="lower right")

plt.tight_layout()
plt.show()


ada = AdaBoostClassifier(estimator=rf,
                         n_estimators=500, 
                         learning_rate=0.1,
                         random_state=1)
ada.fit(X_train, y_train)

ada_cf, ada_acc, ada_precision, ada_recall, ada_f1, ada_auc = output(ada,X_test, y_test)
print("AdaBoost:")
print("Confusion Matrix:")
print(ada_cf)
print("Accuracy:", ada_acc)
print("Precision:", ada_precision)
print("Recall:", ada_recall)
print("F1 Score:", ada_f1)
print("AUC:", ada_auc)

fig = plt.figure(figsize=(7, 5))
mean_tpr = 0.0
mean_fpr = np.linspace(0, 1, 100)
all_tpr = []

# 遍歷每個交叉驗證折疊
for i, (train, test) in enumerate(cv):
    # 訓練模型並獲取預測概率
    probas = rf.fit(X_train2.iloc[train],
                         y_train[train]).predict_proba(X_train2.iloc[test])
    
    # 計算 ROC 曲線的指標
    fpr, tpr, thresholds = roc_curve(y_train[test],
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
plt.xlabel('AdaBoost :False positive rate')
plt.ylabel('AdaBoost :True positive rate')
plt.legend(loc="lower right")

plt.tight_layout()
plt.show()