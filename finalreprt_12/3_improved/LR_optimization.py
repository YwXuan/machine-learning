import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, learning_curve
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix,precision_score, recall_score, f1_score,roc_auc_score,roc_curve,auc
data = pd.DataFrame(   data = pd.read_csv("./processed_dataset.csv"), \
        columns=['brand','aggregateRating/reviewCount','offers/price','depth','width','aggregateRating/ratingValue'])
X = data[['brand','aggregateRating/reviewCount','offers/price','depth','width']]
y = data['aggregateRating/ratingValue']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=1)
lr_model = Pipeline([
    ('poly', PolynomialFeatures(degree=2)), ('Standard', StandardScaler()),('PCA', PCA(n_components=5)),
    ('lr', LogisticRegression(penalty='l2', max_iter=100, solver='lbfgs', C=20.0))
])  #LR模型
rf_model = Pipeline([
    ('poly', PolynomialFeatures(degree=2)),('Standard', StandardScaler()),
    ('PCA', PCA(n_components=5)),('rf', RandomForestClassifier(n_estimators=100, random_state=1))
])  #隨機森林
model = VotingClassifier(estimators=[('lr', lr_model), ('rf', rf_model)],voting='soft')
# 訓練集成模型
model.fit(X_train, y_train)
y_pred_ensemble = model.predict(X_test)
confmat_ensemble = confusion_matrix(y_true=y_test, y_pred=y_pred_ensemble)
y_proba_ensemble = model.predict_proba(X_test)
auc_score_ensemble = roc_auc_score(y_test, y_proba_ensemble, multi_class='ovr')
print('LR & RF Model Confusion Matrix:')
print(confmat_ensemble)
print('------------------------')
print('LR & RF Model Test Accuracy: %.3f' % model.score(X_test, y_test))
print('LR & RF Model Precision: %.3f' % precision_score(y_true=y_test, y_pred=y_pred_ensemble, average='weighted', zero_division=1))
print('LR & RF Model Recall: %.3f' % recall_score(y_true=y_test, y_pred=y_pred_ensemble, average='weighted'))
print('LR & RF Model F1: %.3f' % f1_score(y_true=y_test, y_pred=y_pred_ensemble, average='weighted'))
print('LR & RF Model AUC: %.3f' % auc_score_ensemble)
# 繪製 ROC 曲線
plt.figure(figsize=(10, 7))
for i in range(len(model.classes_)):
    fpr, tpr, _ = roc_curve(y_test, y_proba_ensemble[:, i], pos_label=model.classes_[i])
    plt.plot(fpr, tpr, label=f'class {model.classes_[i]} (AUC = {auc(fpr, tpr):.2f})')
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('LR & RF Model ROC Curve')
plt.legend(loc='best')
plt.show()

train_sizes, train_scores, test_scores =\
      learning_curve(model, X_train, y_train, \
                       cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10))
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