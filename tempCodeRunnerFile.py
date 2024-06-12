Accuracy: %.3f' % svm.score(X_test, y_test))
print('SVM-Precision: %.3f' % precision_score(y_true=y_test, y_pred=y_pred,average='weighted', zero_division=1))
print('SVM-Recall: %.3f' % recall_score(y_true=y_test, y_pred=y_pred ,average='weighted'))
print('SVM-F1: %.3f' % f1_score(y_true=y_test, y_pred=y_pred ,average='weighted'))

