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
