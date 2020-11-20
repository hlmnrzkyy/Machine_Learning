from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

iris = datasets.load_iris()
atribut = iris.data
label = iris.target

# bagi dataset menjadi train set dan test set
X_train, X_test, y_train, y_test = train_test_split(
    atribut, label, test_size=0.2)

decision_tree = tree.DecisionTreeClassifier()
model_pertama = decision_tree.fit(X_train, y_train)
print("---------------------------------")
print(model_pertama.score(X_test, y_test))
print("---------------------------------")

# membuat objek PCA dengan 4 principal component
pca = PCA(n_components=4)

# mengaplikasikan PCA pada dataset
pca_attributes = pca.fit_transform(X_train)

# melihat variance dari setiap atribut
print("---------------------------------")
print(pca.explained_variance_ratio_)
print("---------------------------------")

# PCA dengan 2 principal component
pca = PCA(n_components = 2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.fit_transform(X_test)

# uji akurasi classifier
model2 = decision_tree.fit(X_train_pca, y_train)
print("---------------------------------")
print(model2.score(X_test_pca, y_test))
print("---------------------------------")

# membuat grafik perbandingan antara 1 pca, 2 pca, 3 pca, dan 4 pca
X1 = []
X2 = []
X3 = []
X4 = []
for j in range(1, 5):
    for i in range(0, 100):

        atribut = iris.data
        label = iris.target

        # bagi dataset menjadi train set dan test set
        X_train, X_test, y_train, y_test = train_test_split(
            atribut, label, test_size=0.2)

        pca = PCA(n_components=j)
        pca_attributes = pca.fit_transform(X_train)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.fit_transform(X_test)

        # uji akurasi classifier
        model = decision_tree.fit(X_train_pca, y_train)
        score = model.score(X_test_pca, y_test)

        # menyimpan hasil pada list X dan Y
        if j == 1:
            X1.append(score)
        elif j == 2:
            X2.append(score)
        elif j == 3:
            X3.append(score)
        elif j == 4:
            X4.append(score)

plt.figure(1)
plt.boxplot([X1, X2, X3, X4], labels = ["1 PCA", "2 PCA", "3 PCA", "4 PCA",])
plt.title("Compare boxplot antara jumlah PCA yang digunakan \n dan nilai akurasi yang dihasilkan dari decision tree")
plt.show()



