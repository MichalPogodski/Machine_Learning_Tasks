from sklearn import datasets, cluster, decomposition
import matplotlib.pyplot as plt


def plot_irs(X, y):
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=y)

  
def plot_irs_3d(X, y):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y)



def task_1():
    iris = datasets.load_iris()
    X, y = iris.data, iris.target

    # 3 clusters
    kmeans = cluster.KMeans(n_clusters=3)
    kmeans.fit(X)
    kmeans_3 = kmeans.labels_
    print(kmeans_3)
    print(y)

    # all clusters
    kmeans = cluster.KMeans()
    kmeans.fit(X)
    kmeans_default = kmeans.predict(X)
    print(kmeans_default)
    print(y)


    plot_irs_3d(X, y)
    plot_irs_3d(X, kmeans_3)
    plot_irs_3d(X, kmeans_default)
    plt.show()


def task_2():
    iris = datasets.load_iris()
    X, y = iris.data, iris.target

    clusters = range(2, 15)
    inertias = []
    for n in clusters:
        kmeans = cluster.KMeans(n_clusters=n).fit(X)
        inertias.append(kmeans.inertia_)

    plt.plot(clusters, inertias, '-o', color='black')
    plt.xlabel('number of clusters, k')
    plt.xticks(clusters)
    plt.show()


def task_3():
    iris = datasets.load_iris()
    X, y = iris.data, iris.target

    pca = decomposition.PCA(n_components=2)
    pca.fit(X)
    X_r = pca.transform(X)

    plot_irs(X, y)
    plot_irs(X_r, y)
    plt.show()

if __name__ == '__main__':
    # task_1()
    # task_2()
    task_3()
