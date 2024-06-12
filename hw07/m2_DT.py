from data_processing import preprocess_data
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import precision_score, recall_score, f1_score,accuracy_score,roc_auc_score
from sklearn.ensemble import BaggingClassifier


data = pd.DataFrame(preprocess_data("best_buy_laptops_2024.csv"), \
        columns=['brand_encoded','model_encoded','satisfaction_level','aggregateRating/ratingValue'\
                ,'aggregateRating/reviewCount','offers/price','depth','features/0/description_encoded',\
                'features/1/description_encoded'])
X = data[['brand_encoded','model_encoded','aggregateRating/ratingValue','aggregateRating/reviewCount',\
          'offers/price','depth','features/0/description_encoded','features/1/description_encoded']]
y = data['satisfaction_level']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=1)

dt = make_pipeline(StandardScaler(),
                    PCA(n_components=2),
                    DecisionTreeClassifier(criterion='gini', 
                                    max_depth=4, 
                                    random_state=1))

dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)

y_pred = dt.predict(X_test)
confmat = confusion_matrix(y_true=y_test, y_pred=y_pred )

y_proba =dt.predict_proba(X_test)
auc = roc_auc_score(y_test, y_proba, multi_class='ovr')

print('DT-confusion matrix:')
print(confmat)
print('------------------------')
print('DT-Test Accuracy: %.3f' % dt.score(X_test, y_test))
print('DT-Precision: %.3f' % precision_score(y_true=y_test, y_pred=y_pred,average='weighted', zero_division=1))
print('DT-Recall: %.3f' % recall_score(y_true=y_test, y_pred=y_pred ,average='weighted'))
print('DT-F1: %.3f' % f1_score(y_true=y_test, y_pred=y_pred ,average='weighted'))
print('DT-AUC: %.3f' % auc)


bag = BaggingClassifier(estimator=dt,
                        n_estimators=500, 
                        max_samples=0.8, 
                        max_features=0.5, 
                        bootstrap=True, 
                        bootstrap_features=False, 
                        n_jobs=1, 
                        random_state=1)
X1_train, X1_test, y1_train, y1_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=1)

bag = bag.fit(X1_train, y1_train)
y1_train_pred = bag.predict(X1_train)
y1_test_pred = bag.predict(X1_test)
confmat1 = confusion_matrix(y_true=y1_test, y_pred=y1_test_pred )
bag_train = accuracy_score(y1_train, y1_train_pred) 
bag_test = accuracy_score(y1_test, y1_test_pred) 
y1_pred = bag.predict(X1_test)

confmat1 = confusion_matrix(y_true=y1_test, y_pred=y1_pred )
y1_proba = bag.predict_proba(X_test)
auc1 = roc_auc_score(y1_test, y1_proba, multi_class='ovr')

print('BG-confusion matrix:')
print(confmat1)
print('------------------------')
print('BG-Test Accuracy: %.3f' % bag.score(X1_test, y1_test))
print('BG-Precision: %.3f' % precision_score(y_true=y1_test, y_pred=y1_pred,average='weighted', zero_division=1))
print('BG-Recall: %.3f' % recall_score(y_true=y1_test, y_pred=y1_pred ,average='weighted'))
print('BG-F1: %.3f' % f1_score(y_true=y1_test, y_pred=y1_pred ,average='weighted'))
print('BG-AUC: %.3f' % auc1)
