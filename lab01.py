import pickle
from sklearn import datasets, svm
import matplotlib.pyplot as plt
import numpy as np

def task_0():
    digits = datasets.load_digits()
    # print(digits)
    # print(digits.DESCR)

    plt.imshow(digits.images[0])
    # plt.show()
    # print(digits.target_names)

    clf = svm.SVC()
    clf.fit(digits.data[0:10], digits.target[0:10])
    print(clf.predict([digits.data[0]]))
    pickle.dump(clf, open('./clf.p', 'wb'))
    pickle.load(open('./clf.p', 'rb'))


def task_4():
    faces = datasets.fetch_olivetti_faces()
    print(faces) #, '\n', faces.DESCR, '\n', faces.data, '\n', faces.target)
    X, y = datasets.fetch_olivetti_faces(return_X_y=True)
    print(y)


def task_5():
    from sklearn.datasets import load_boston
    boston = load_boston()
    print(boston)
    print(boston.DESCR)
    print(boston.data)
    print(boston.target)
    print(boston['feature_names'])


def task_6():
    x, y = datasets.make_classification(
        n_samples=100,
        n_features=3,
        n_informative=3, n_redundant=0, n_repeated=0,
        n_classes=4,
        n_clusters_per_class=1,
        class_sep=5.0,
        flip_y=0.0
    )
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x[:,0], x[:,1], x[:,2], c=y, marker='o')

    ax.set_xlabel('X label')
    ax.set_ylabel('Y label')
    ax.set_zlabel('Z label')

    plt.show()



def task_7():
    d = datasets.fetch_openml(data_id=40536, as_frame=True)
    print(type(d))

    diabetes = datasets.load_diabetes(as_frame=True)
    print(diabetes.frame.head(5))


def final_Boss():

    def regressor_9000(x: float) -> float:
        if x >= 4.0:
            return 8.0
        else:
            return x*2


    data = np.loadtxt('./battery_data.csv', delimiter=',')
    print(data)

    x = data[:, 0]
    y = data[:, 1]

    y_predicted = []
    for single_data in x:
        y_predicted.append(regressor_9000(single_data))

    plt.scatter(x, y)
    plt.scatter(x, y_predicted, marker='*', c='r')
    plt.show()

if __name__ == '__main__':
    # task_0()
    # task_4()
    # task_5()
    # task_6()
    # task_7()
    final_Boss()