from sklearn import datasets, model_selection, svm, metrics, ensemble, impute
import matplotlib.pyplot as plt


# Data analyzing and processing
def task_1():
    X, y = datasets.fetch_openml('diabetes', as_frame=True, return_X_y=True)
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, test_size=0.2, random_state=42)

    plt.figure()
    X_train.boxplot()
    X_train.hist()

    # filling missing values
    imputer = impute.SimpleImputer(missing_values=0.0, strategy='mean')
    X_train[['mass']] = imputer.fit_transform(X_train[['mass']])
    X_train[['skin']] = imputer.fit_transform(X_train[['skin']])

    plt.figure()
    X_train.boxplot()
    X_train.hist()

    # marking outliers
    isolation_forest = ensemble.IsolationForest(contamination=0.05)
    isolation_forest.fit(X_train)
    y_predicted_outliers = isolation_forest.predict(X_test)
    print(y_predicted_outliers)


    clf_svm = svm.SVC()
    clf_svm.fit(X_train, y_train)
    y_predicted_svm = clf_svm.predict(X_test)
    # print(metrics.classification_report(y_test, y_predicted_svm))

    clf_rf = ensemble.RandomForestClassifier()
    clf_rf.fit(X_train, y_train)
    y_predicted_rf = clf_rf.predict(X_test)
    # print(metrics.classification_report(y_test, y_predicted_rf))

    plt.show()

if __name__ == '__main__':
    task_1()

