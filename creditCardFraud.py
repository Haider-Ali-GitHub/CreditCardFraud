# IMPORTING PACKAGES

import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

# IMPORTING DATA
credit_data = pd.read_csv('uploads/creditcard.csv')
credit_data.drop('Time', axis=1, inplace=True)

# EDA
total_transactions = len(credit_data)
legit_transactions = len(credit_data[credit_data.Class == 0])
fraud_transactions = len(credit_data[credit_data.Class == 1])
fraud_ratio = round(fraud_transactions / legit_transactions * 100, 2)

# DATA SPLIT
features = credit_data.drop('Class', axis=1).values
labels = credit_data['Class'].values
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2, random_state=0)

# MODELS
decision_tree = DecisionTreeClassifier(max_depth=4, criterion='entropy')
k_nearest_neighbors = KNeighborsClassifier(n_neighbors=5)
logistic_reg = LogisticRegression()
support_vector_machine = SVC()
random_forest = RandomForestClassifier(max_depth=4)
extreme_gradient_boosting = XGBClassifier(max_depth=4)

# TRAINING
decision_tree.fit(features_train, labels_train)
k_nearest_neighbors.fit(features_train, labels_train)
logistic_reg.fit(features_train, labels_train)
support_vector_machine.fit(features_train, labels_train)
random_forest.fit(features_train, labels_train)
extreme_gradient_boosting.fit(features_train, labels_train)

# PREDICTIONS
decision_tree_predictions = decision_tree.predict(features_test)
knn_predictions = k_nearest_neighbors.predict(features_test)
logistic_reg_predictions = logistic_reg.predict(features_test)
svm_predictions = support_vector_machine.predict(features_test)
random_forest_predictions = random_forest.predict(features_test)
xgboost_predictions = extreme_gradient_boosting.predict(features_test)

# EVALUATION
print(f'Accuracy of Decision Tree: {accuracy_score(labels_test, decision_tree_predictions)}')
print(f'Accuracy of KNN: {accuracy_score(labels_test, knn_predictions)}')
print(f'Accuracy of Logistic Regression: {accuracy_score(labels_test, logistic_reg_predictions)}')
print(f'Accuracy of SVM: {accuracy_score(labels_test, svm_predictions)}')
print(f'Accuracy of Random Forest: {accuracy_score(labels_test, random_forest_predictions)}')
print(f'Accuracy of XGBoost: {accuracy_score(labels_test, xgboost_predictions)}')

print(f'F1 Score of Decision Tree: {f1_score(labels_test, decision_tree_predictions)}')
print(f'F1 Score of KNN: {f1_score(labels_test, knn_predictions)}')
print(f'F1 Score of Logistic Regression: {f1_score(labels_test, logistic_reg_predictions)}')
print(f'F1 Score of SVM: {f1_score(labels_test, svm_predictions)}')
print(f'F1 Score of Random Forest: {f1_score(labels_test, random_forest_predictions)}')
print(f'F1 Score of XGBoost: {f1_score(labels_test, xgboost_predictions)}')

# CONFUSION MATRIX PLOTTING
def plot_confusion_matrix(cm, classes, title):
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()

# CONFUSION MATRICES FOR EACH MODEL
for model, predictions, title in zip(
    [decision_tree, k_nearest_neighbors, logistic_reg, support_vector_machine, random_forest, extreme_gradient_boosting],
    [decision_tree_predictions, knn_predictions, logistic_reg_predictions, svm_predictions, random_forest_predictions, xgboost_predictions],
    ['Decision Tree', 'KNN', 'Logistic Regression', 'SVM', 'Random Forest', 'XGBoost']
):
    conf_matrix = confusion_matrix(labels_test, predictions)
    plt.figure()
    plot_confusion_matrix(conf_matrix, classes=['Legit(0)', 'Fraud(1)'], title=f'Confusion Matrix of {title}')
    plt.savefig(f'{title}_confusion_matrix.png')
    plt.show()
