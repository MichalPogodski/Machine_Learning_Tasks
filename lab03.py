from sklearn import datasets, model_selection, preprocessing, svm, linear_model, tree, ensemble, metrics
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions

def task_1():

    def plot_irs(X):
        plt.figure()
        plt.scatter(X[:, 0], X[:, 1])
        plt.axvline(x=0)
        plt.axhline(y=0)
        plt.title('Iris sepal features')
        plt.xlabel('sepal length (cm)')
        plt.ylabel('sepal width (cm)')


    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    print('count y: ', np.bincount(y))
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        iris.data, iris.target, test_size=0.25, stratify=y)
    print('count y_train: ', np.bincount(y_train))
    print('count y_test: ', np.bincount(y_test))

    plot_irs(X)


    min_max_scaler = preprocessing.MinMaxScaler()
    min_max_scaler.fit(X_train)
    min_max_scaler.transform(X)
    X_min_max_scaled = min_max_scaler.transform(X)

    plot_irs(X_min_max_scaled)
    plt.show()



def main_task():
    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    X = X[:, [0, 1]]
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42)

    min_max_scaler = preprocessing.MinMaxScaler()
    min_max_scaler.fit(X_train)
    X_train = min_max_scaler.transform(X_train)
    X_test = min_max_scaler.transform(X_test)

    clf_svm = svm.SVC(random_state=42)
    clf_svm.fit(X_train, y_train)
    acc_svm = metrics.accuracy_score(y_test, clf_svm.predict(X_test))
    print('svm acc: ', acc_svm)

    clf_linear = linear_model.LogisticRegression(random_state=42)
    clf_linear.fit(X_train, y_train)
    acc_linear = metrics.accuracy_score(y_test, clf_linear.predict(X_test))
    print('linear acc: ', acc_linear)

    clf_tree = tree.DecisionTreeClassifier(random_state=42)
    clf_tree.fit(X_train, y_train)
    acc_tree = metrics.accuracy_score(y_test, clf_tree.predict(X_test))
    print('tree acc: ', acc_tree)

    clf_rf = ensemble.RandomForestClassifier(random_state=42)
    clf_rf.fit(X_train, y_train)
    acc_rf = metrics.accuracy_score(y_test, clf_rf.predict(X_test))
    print('rf acc: ', acc_rf)

    param_grid = [
        {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
        {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['linear']},
    ]

    clf_gs = model_selection.GridSearchCV(estimator=svm.SVC(), param_grid=param_grid)
    clf_gs.fit(X_train, y_train)
    # print(clf_gs.cv_results_)
    acc_gs = metrics.accuracy_score(y_test, clf_gs.predict(X_test))
    print('gs acc: ', acc_gs)



    plt.figure()
    plot_decision_regions(X_train, y_train, clf=clf_svm, legend=2)
    plt.figure()
    plot_decision_regions(X_train, y_train, clf=clf_linear, legend=2)
    plt.figure()
    plot_decision_regions(X_train, y_train, clf=clf_tree, legend=2)
    plt.figure()
    plot_decision_regions(X_train, y_train, clf=clf_rf, legend=2)
    plt.figure()
    plot_decision_regions(X_train, y_train, clf=clf_gs, legend=2)
    plt.show()



if __name__ == '__main__':
    # task_1()
    main_task()

