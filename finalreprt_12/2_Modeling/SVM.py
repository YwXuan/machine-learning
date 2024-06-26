import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix,precision_score, recall_score, f1_score,roc_auc_score,roc_curve,auc
data = pd.DataFrame(   data = pd.read_csv("./processed_dataset.csv"), \
        columns=['brand','aggregateRating/reviewCount','offers/price','depth','width','aggregateRating/ratingValue'])
X = data[['brand','aggregateRating/reviewCount','offers/price','depth','width']]
y = data['aggregateRating/ratingValue']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=1)
svm = make_pipeline(StandardScaler(),SVC(kernel='poly', degree=3, gamma='scale', C=10.0, probability=True))
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)
confmat = confusion_matrix(y_true=y_test, y_pred=y_pred )
y_proba =svm.predict_proba(X_test)
print('SVM-confusion matrix:')
print(confmat)
print('------------------------')
print('SVM-Test Accuracy: %.3f' % svm.score(X_test, y_test))
print('SVM-Precision: %.3f' % precision_score(y_true=y_test, y_pred=y_pred,average='weighted', zero_division=1))
print('SVM-Recall: %.3f' % recall_score(y_true=y_test, y_pred=y_pred ,average='weighted'))
print('SVM-F1: %.3f' % f1_score(y_true=y_test, y_pred=y_pred ,average='weighted'))
print('SVM-AUC: %.3f' % roc_auc_score(y_test, y_proba, multi_class='ovr'))  
plt.figure(figsize=(10,7))
for i in range(len(svm.classes_)):
    fpr, tpr, _ = roc_curve(y_test, y_proba[:, i], pos_label=svm.classes_[i])
    plt.plot(fpr,tpr,label=f'class {svm.classes_[i]} (AUC = {auc(fpr, tpr):.2f})')
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('SVM- ROC Curve')
plt. legend(loc='best')
plt.show()

y_train_pred = svm.predict(X_train)
y_test_pred = svm.predict(X_test)
plt.figure(figsize=(10, 7))
plt.scatter(y_train_pred, y_train_pred - y_train,
            c='lightblue', marker='o',
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

from sklearn.model_selection import learning_curve
import numpy as np

train_sizes, train_scores, test_scores = \
    learning_curve(svm, X_train, y_train, cv=5, n_jobs=-1, \
                   train_sizes=np.linspace(0.1, 1.0, 10))
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

plt.xlabel("Training")
plt.ylabel("Score")
plt.title("Learning Curve")
plt.legend(loc="best")
plt.grid()
plt.show()