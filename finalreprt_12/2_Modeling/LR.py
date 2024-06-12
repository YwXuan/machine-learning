import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix,precision_score, recall_score, f1_score,roc_auc_score,roc_curve,auc
data = pd.DataFrame(   data = pd.read_csv("./processed_dataset.csv"), \
        columns=['brand','aggregateRating/reviewCount','offers/price','depth','width','aggregateRating/ratingValue'])
X = data[['brand','aggregateRating/reviewCount','offers/price','depth','width']]
y = data['aggregateRating/ratingValue']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=1)
lg = Pipeline([('poly',PolynomialFeatures(degree=2)),('Standard',StandardScaler()),
    ('PCA',PCA(n_components=5)),('lg-reg',LogisticRegression(penalty='l2', max_iter=100,solver='lbfgs',C=20.0))])
lg.fit(X_train, y_train)
y_pred = lg.predict(X_test)
confmat = confusion_matrix(y_true=y_test, y_pred=y_pred )
y_proba =lg.predict_proba(X_test)
auc_score = roc_auc_score(y_test, y_proba, multi_class='ovr')
print('LR-confusion matrix:')
print(confmat)
print('------------------------')
print('LR-Test Accuracy: %.3f' % lg.score(X_test, y_test))
print('LR-Precision: %.3f' % precision_score(y_true=y_test, y_pred=y_pred,average='weighted', zero_division=1))
print('LR-Recall: %.3f' % recall_score(y_true=y_test, y_pred=y_pred ,average='weighted'))
print('LR-F1: %.3f' % f1_score(y_true=y_test, y_pred=y_pred ,average='weighted'))
print('LR-AUC: %.3f' % auc_score)
plt.figure(figsize=(10,7))
for i in range(len(lg.classes_)):
    fpr, tpr, _ = roc_curve(y_test, y_proba[:, i], pos_label=lg.classes_[i])
    plt.plot(fpr,tpr,label=f'class {lg.classes_[i]} (AUC = {auc(fpr, tpr):.2f})')
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('LG- ROC Curve')
plt. legend(loc='best')
plt.show()

y_train_pred = lg.predict(X_train)
y_test_pred = lg.predict(X_test)
plt.figure(figsize=(10, 7))
plt.scatter(y_train_pred, y_train_pred - y_train,
            c='blue', marker='o',
            label='Training data',s=100, alpha=1)
plt.scatter(y_test_pred, y_test_pred - y_test,
            c='lightgreen', marker='s',
            label='Test data',s=100, alpha=0.5)
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0.5, xmin=min(min(y_train_pred)-0.5, min(y_test_pred))-0.5, xmax=max(max(y_train_pred)+0.5, max(y_test_pred)+0.5), color='black', lw=2)
plt.xlim([min(min(y_train_pred)-0.5, min(y_test_pred))-0.5, max(max(y_train_pred)+0.5, max(y_test_pred)+0.5)])
plt.tight_layout()

plt.show()

train_sizes, train_scores, test_scores =\
      learning_curve(lg, X_train, y_train, cv=5,\
                      n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10))
train_scores_mean = train_scores.mean(axis=1)
train_scores_std = train_scores.std(axis=1)
test_scores_mean = test_scores.mean(axis=1)
test_scores_std = test_scores.std(axis=1)
plt.figure(figsize=(10, 7))
plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.1,
                 )
plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.1)
plt.plot(train_sizes, train_scores_mean, 'o-', 
         label="Training")
plt.plot(train_sizes, test_scores_mean, 'v-', 
         label="Cross-validation")

plt.xlabel("LR Training")
plt.ylabel("LR Score")
plt.title("LR Learning Curve")
plt.legend(loc="best")
plt.grid()
plt.show()