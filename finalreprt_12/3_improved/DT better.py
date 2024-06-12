import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
file_path = 'processed_dataset.csv'
data = pd.read_csv(file_path)
features = ['brand', 'aggregateRating/reviewCount', 'offers/price', 'depth', 'width']
target = 'aggregateRating/ratingValue'
X = data[features]
y = data[target]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
# 定義模型及其參數範圍
models = {
    'Decision Tree': {
        'model': DecisionTreeClassifier(),
        'params': {
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 10, 20]
        }
    },
    'Random Forest': {
        'model': RandomForestClassifier(),
        'params': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 10, 20]
        }
    },
    'Gradient Boosting': {
        'model': GradientBoostingClassifier(),
        'params': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 10]
        }
    }
}
# 尋找最佳模型
best_models = {}
for name, model_info in models.items():
    grid_search = GridSearchCV(model_info['model'], model_info['params'], cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    best_models[name] = grid_search.best_estimator_
    print(f"{name} best params: {grid_search.best_params_}")
# 選擇性能最好的模型進行評估
best_model_name = max(best_models, key=lambda name: accuracy_score(y_test, best_models[name].predict(X_test)))
best_model = best_models[best_model_name]
# 模型評估
y_pred = best_model.predict(X_test)
y_pred_prob = best_model.predict_proba(X_test)
cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_prob, multi_class='ovr')
print(f"Best Model: {best_model_name}")
print(f"Confusion Matrix:\n{cm}")
print(f"Accuracy: {accuracy}")
print(f"Classification Report:\n{report}")
print(f"AUC: {auc}")
fpr, tpr = {}, {}
plt.figure()
for i, color in zip(range(len(best_model.classes_)), ['blue', 'red', 'green']):
    fpr[i], tpr[i], _ = roc_curve(y_test, y_pred_prob[:, i], pos_label=best_model.classes_[i])
    plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f'ROC fold {best_model.classes_[i]} (area = {roc_auc_score(y_test == best_model.classes_[i], y_pred_prob[:, i]):.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.show()

y_train_pred = best_model.predict(X_train)
y_test_pred = best_model.predict(X_test)

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