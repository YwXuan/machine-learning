import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline

filePath = 'processed_dataset.csv'
dfCleaning = pd.read_csv(filePath)

X = dfCleaning.drop(columns=['aggregateRating/ratingValue'])
y = dfCleaning['aggregateRating/ratingValue']

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
rf_model = RandomForestClassifier(random_state=42, oob_score=True)
rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)
y_pred_prob = rf_model.predict_proba(X_test)

cm = confusion_matrix(y_test, y_pred)
print('Confusion matrix')
print(cm)
print('----------------')
accuracy = accuracy_score(y_test, y_pred)
print('Test Accuracy:', accuracy)
print('----------------')

try:
    auc = roc_auc_score(y_test, y_pred_prob, multi_class='ovr')
    print('ROC AUC Score:', auc)
except ValueError as e:
    print(f'Error calculating ROC AUC Score: {e}')
print('----------------')

ps = precision_score(y_test, y_pred, average='weighted', zero_division=1)
rs = recall_score(y_test, y_pred, average='weighted')
f1sc = f1_score(y_test, y_pred, average='weighted')
print('Precision Score:', ps)
print('Recall Score:', rs)
print('F1 Score:', f1sc)

plt.figure(figsize=(10, 7))
for i in range(len(rf_model.classes_)):
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob[:, i], pos_label=rf_model.classes_[i])
    plt.plot(fpr, tpr, label=f'Class {rf_model.classes_[i]} (AUC={auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('RF ROC Curve')
plt.legend(loc='best')
plt.show()

y_test_pred = rf_model.predict(X_test)
y_train_pred = rf_model.predict(X_train)

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

train_sizes, train_scores, validation_scores = learning_curve(
    estimator=rf_model,
    X=X,
    y=y,
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)

validation_scores_mean = np.mean(validation_scores, axis=1)
validation_scores_std = np.std(validation_scores, axis=1)

plt.figure()
plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Training score')
plt.plot(train_sizes, validation_scores_mean, 'v-', color='orange', label='Validation')

plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color='r')
plt.fill_between(train_sizes, validation_scores_mean - validation_scores_std, validation_scores_mean + validation_scores_std, alpha=0.1, color='orange')

plt.title('Learning Curve')
plt.xlabel('RF')
plt.ylabel('Score')
plt.legend(loc='best')
plt.grid()
plt.show()
