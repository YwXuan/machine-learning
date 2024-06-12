from data_processing import preprocess_data 
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA



data = pd.DataFrame(preprocess_data("best_buy_laptops_2024.csv"), columns=['brand_encoded','model_encoded','satisfaction_level','aggregateRating/ratingValue','aggregateRating/reviewCount','offers/price','depth','features/0/description_encoded','features/1/description_encoded'])


X = data[['brand_encoded','model_encoded','aggregateRating/ratingValue','aggregateRating/reviewCount','offers/price','depth','features/0/description_encoded','features/1/description_encoded']]
y = data['satisfaction_level']

# 標準化特徵
# scaler = StandardScaler()
# X_std = scaler.fit_transform(X)
# X_train_std, X_test_std, y_train, y_test = train_test_split(X_std, y, test_size=0.2, random_state=42)
# X_combined_std = np.vstack((X_train_std, X_test_std))
# y_combined = np.hstack((y_train, y_test))


# norm_data=pd.concat([X,y] , axis=1)
# x_train, x_test, y_train, y_test=train_test_split( X ,y,test_size=0.2,random_state=42)
# y_combined = np.hstack((y_train, y_test))

# X_train_std = scaler.transform(x_train)
# X_test_std = scaler.transform(x_test)
# X_combined_std = np.vstack((X_train_std, X_test_std))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=1)

lg = make_pipeline(StandardScaler(),
                        PCA(n_components=2),
                        LogisticRegression(random_state=1, solver='lbfgs'))

lg.fit(X_train, y_train)
y_pred = lg.predict(X_test)

y_pred = lg.predict(X_test)
confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)

fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')

plt.xlabel('LG-Predicted label')
plt.ylabel('LG-True label')

plt.tight_layout()
#plt.savefig('images/06_09.png', dpi=300)
plt.show()

print('LG-Test Accuracy: %.3f' % lg.score(X_test, y_test))
print('LG-Precision: %.3f' % precision_score(y_true=y_test, y_pred=y_pred,average='weighted', zero_division=1))
print('LG-Recall: %.3f' % recall_score(y_true=y_test, y_pred=y_pred ,average='weighted'))
print('LG-F1: %.3f' % f1_score(y_true=y_test, y_pred=y_pred ,average='weighted'))


# SVM模型

svm = make_pipeline(StandardScaler(),
                         SVC(random_state=1))

svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)

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

print('SVM-Test Accuracy: %.3f' % lg.score(X_test, y_test))
print('SVM-Precision: %.3f' % precision_score(y_true=y_test, y_pred=y_pred,average='weighted', zero_division=1))
print('SVM-Recall: %.3f' % recall_score(y_true=y_test, y_pred=y_pred ,average='weighted'))
print('SVM-F1: %.3f' % f1_score(y_true=y_test, y_pred=y_pred ,average='weighted'))