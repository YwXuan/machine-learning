import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt

data = pd.read_csv('processed_dataset.csv')
X = data.drop('aggregateRating/ratingValue', axis=1)
y = data['aggregateRating/ratingValue']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# 使用GridSearchCV調整超參數
param_grid = {'n_neighbors': np.arange(1, 31), 'weights': ['uniform', 'distance']}
knn = KNeighborsClassifier()
knn_cv = GridSearchCV(knn, param_grid, cv=5)
knn_cv.fit(X_train, y_train)
print(f"Best parameters: {knn_cv.best_params_}")

# 使用最佳參數訓練模型
best_knn = knn_cv.best_estimator_
best_knn.fit(X_train, y_train)
y_pred = best_knn.predict(X_test)
y_pred_prob = best_knn.predict_proba(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
roc_auc = roc_auc_score(y_test, y_pred_prob, multi_class='ovr')

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("AUC:", roc_auc)

# 繪製ROC 曲線
plt.figure(figsize=(10, 7))
for i in range(len(best_knn.classes_)):
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob[:, i], pos_label=best_knn.classes_[i])
    plt.plot(fpr, tpr, label=f'Class {best_knn.classes_[i]} (AUC = {auc(fpr, tpr):.2f})')
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='best')
plt.show()

# 繪製殘差圖
y_train_pred = knn_cv.predict(X_train)
y_test_pred = knn_cv.predict(X_test)

plt.figure(figsize=(10, 7))
plt.scatter(y_train_pred, y_train_pred - y_train, c='blue', marker='o', label='Training data', s=100, alpha=1)
plt.scatter(y_test_pred, y_test_pred - y_test, c='lightgreen', marker='s', label='Test data', s=100, alpha=0.5)
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=min(min(y_train_pred)-0.5, min(y_test_pred))-0.5, xmax=max(max(y_train_pred)+0.5, max(y_test_pred)+0.5), color='black', lw=2)
plt.xlim([min(min(y_train_pred)-0.5, min(y_test_pred))-0.5, max(max(y_train_pred)+0.5, max(y_test_pred)+0.5)])
plt.tight_layout()
plt.show()

# 繪製學習曲線
train_sizes, train_scores, test_scores = learning_curve(knn_cv, X, y, cv=5, scoring='accuracy', 
                                                        n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10))

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.figure(figsize=(10, 7))
plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Training score')
plt.plot(train_sizes, test_scores_mean, 'o-', color='g', label='Cross-validation score')

plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std,
                  alpha=0.1, color='r')
plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std,
                  alpha=0.1, color='g')

plt.title('Learning Curve')
plt.xlabel('Training')
plt.ylabel('Score')
plt.legend(loc='best')
plt.grid()
plt.show()
