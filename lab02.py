from sklearn import tree, linear_model, metrics, datasets, model_selection
import matplotlib.pyplot as plt
import numpy as np


def task_1(): #DecisionTreeClassifier logic AND
    X = [[0, 0],
         [0, 1],
         [1, 0],
         [1, 1]]
    y = [0, 0, 0, 1]

    clf = tree.DecisionTreeClassifier()
    clf.fit(X, y)

    print(clf.predict([[1, 1]]))



def task_2(): #DecisionTreeClassifier logic OR
    X = [[0, 0],
         [0, 1],
         [1, 0],
         [1, 1]]
    y = [0, 1, 1, 1]

    clf = tree.DecisionTreeClassifier()
    clf.fit(X, y)

    print(clf.predict([[1, 1]]))

    tree.plot_tree(clf, feature_names=['x1', 'x2'], filled=True)
    plt.show()




def task_3(): #DecisionTreeClassifier BUING A CAR
    dict_brand = {'VW': 0, 'Ford': 1, 'Opel': 2}
    dict_damaged = {'yes': 0, 'no': 1}

    X = [
        ['Opel', 250000, 'yes'],
        ['Ford', 25000, 'no'],
        ['VW', 10000, 'no'],
        ['Ford', 200000, 'no'],
        ['VW', 150000, 'yes'],
        ['Ford', 90000, 'yes'],
        ['Opel', 50000, 'no']
        ]
    for x in X:
        x[0] = dict_brand[x[0]]
        x[2] = dict_damaged[x[2]]

    y = [0, 1, 1, 1, 0, 1, 1]

    clf = tree.DecisionTreeClassifier()
    clf.fit(X, y)

    print(clf.predict(
            [
                [dict_brand['Ford'], 10000, dict_damaged['yes']]
            ]
        ))

    tree.plot_tree(clf, feature_names=['marka', 'przebieg', 'uszkodzenie'],
                  class_names=['zrezygnowac', 'kupic'], filled=True)
    plt.show()




def task_4(): #Confusion matrix for DecisionTreeClassifier on 'digits' dataset
    digits = datasets.load_digits()

    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        digits.data, digits.target, test_size=0.33, random_state=42
    )

    clf = tree.DecisionTreeClassifier()
    clf.fit(X_train, y_train)

    y_predicted = clf.predict(X_test)

    print("Confusion matrix:\n%s" % metrics.confusion_matrix(y_test, y_predicted))
    print("\nClassification report for classifier:\n%s" % (metrics.classification_report(y_test, y_predicted)))
    metrics.plot_confusion_matrix(clf, X_test, y_test)
    plt.show()

    for digit, gt, pred in zip(X_test, y_test, y_predicted):
        if gt != pred:
            print(digit, ' classified as: ', pred, ' while it should be: ', gt, '\n')



def print_regressor_score(y1, y2):
    print('mean_absolute_error: ', metrics.mean_absolute_error(y1, y2))
    print('mean_squared_error: ', metrics.mean_squared_error(y1, y2))
    print('r2_score: ', metrics.r2_score(y1, y2), '\n')


def task_5():
    data = np.loadtxt(fname='./battery_data.csv', delimiter=',')

    X =data[:, 0].reshape(-1, 1)
    y = data[:, 1]

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

    decision_tree_regressor = tree.DecisionTreeRegressor()
    decision_tree_regressor.fit(X_train, y_train)
    y_predicted_decision_tree = decision_tree_regressor.predict(X_test)

    linear_model_regressor = linear_model.LinearRegression()
    linear_model_regressor.fit(X_train, y_train)
    y_predicted_linear_model = linear_model_regressor.predict(X_test)

    print('decision tree:')
    print_regressor_score(y_test, y_predicted_decision_tree)
    print('linear model:')
    print_regressor_score(y_test, y_predicted_linear_model)
    # problem is not linear



if __name__ == '__main__':
    # task_1()
    # task_2()
    # task_3()
    # task_4()
    task_5()

