import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve

#Load the dataset
filePath = 'processed_dataset.csv'
dfCleaning = pd.read_csv(filePath)

#Define features and target
X = dfCleaning.drop(columns=['aggregateRating/ratingValue'])
y = dfCleaning['aggregateRating/ratingValue']

#Scale the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

#Define hyperparameters grid for Random Forest
params = {
    'n_estimators':[200, 300, 400, 500, 600],
    'max_depth': [20, 30, 35, 40, 50, 60],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4, 6],
    'max_features': ['auto', 'sqrt', 'log2']
}

#Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)

#Set up cross-validation
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rf_model = RandomForestClassifier(random_state=42, oob_score=True)
gridRf_model = GridSearchCV(estimator=rf_model, param_grid=params, cv=kf, n_jobs=-1, verbose=2)

#Train the model
gridRf_model.fit(X_train, y_train)
bestRF = gridRf_model.best_estimator_

#Make predictions
y_pred = bestRF.predict(X_test)
y_pred_prob = bestRF.predict_proba(X_test)

#Evaluate the model
cm = confusion_matrix(y_test, y_pred)
print('Confusion matrix:')
print(cm)
print('----------------')
accuracy = accuracy_score(y_test, y_pred)
print('Test Accuracy:', accuracy)
print('----------------')
print('Best Parameters', gridRf_model.best_params_)

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
for i in range(len(bestRF.classes_)):
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob[:, i], pos_label=bestRF.classes_[i])
    plt.plot(fpr, tpr, label=f'Class {bestRF.classes_[i]}')
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('RF ROC Curve')
plt.legend(loc='best')
plt.show()

y_test_pred = bestRF.predict(X_test)
y_train_pred = bestRF.predict(X_train)

plt.figure(figsize=(10, 7))
plt.scatter(y_train_pred, y_train_pred - y_train,
            c='blue', marker='o',
            label='Training data', s=100, alpha=1)
plt.scatter(y_test_pred, y_test_pred - y_test,
            c='lightgreen', marker='s',
            label='Test data', s=100, alpha=0.5)
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=min(min(y_train_pred), min(y_test_pred)), xmax=max(max(y_train_pred), max(y_test_pred)), color='black', lw=2)
plt.xlim([min(min(y_train_pred), min(y_test_pred)) - 0.5, max(max(y_train_pred), max(y_test_pred)) + 0.5])
plt.tight_layout()
plt.show()
