import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import learning_curve
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
file_path = 'Final\processed_dataset.csv'
data = pd.read_csv(file_path)
features = ['brand', 'aggregateRating/reviewCount', 'offers/price', 'depth', 'width']
target = 'aggregateRating/ratingValue'
X = data[features]
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = DecisionTreeClassifier().fit(X_train, y_train)
y_pred = clf.predict(X_test)
y_pred_prob = clf.predict_proba(X_test)
cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_prob, multi_class='ovr')
print(f"Confusion Matrix:\n{cm}")
print(f"Accuracy: {accuracy}")
print(f"Classification Report:\n{report}")
print(f"AUC: {auc}")
fpr, tpr = {}, {}
plt.figure()
for i, color in zip(range(len(clf.classes_)), ['blue', 'red', 'green']):
    fpr[i], tpr[i], _ = roc_curve(y_test, y_pred_prob[:, i], pos_label=clf.classes_[i])
    plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f'ROC fold {clf.classes_[i]} (area = {roc_auc_score(y_test == clf.classes_[i], y_pred_prob[:, i]):.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('DT-False Positive Rate')
plt.ylabel('DT-True Positive Rate')
plt.legend(loc="lower right")
plt.show()

y_train_pred = clf.predict(X_train)
y_test_pred = clf.predict(X_test)

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
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

# 繪製學習曲線
title = "Learning Curves (Decision Tree)"
cv = 5
plot_learning_curve(clf, title, X_train, y_train, cv=cv, n_jobs=-1)

plt.show()