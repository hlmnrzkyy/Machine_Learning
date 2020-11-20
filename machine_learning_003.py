from sklearn import datasets
from sklearn.tree import export_graphviz
from sklearn.tree import DecisionTreeClassifier
from matplotlib import pyplot as plt
from sklearn import tree

# Load iris dataset
iris = datasets.load_iris()
print(iris)

# membuat model Decision Tree
tree_model = DecisionTreeClassifier()

# mendefinisikan atribut dan label pada dataset
x = iris.data
y = iris.target

# melakukan pelatihan model terhadap data
tree_model.fit(x, y)

# prediksi model dengan tree_model.predict([[SepalLength, SepalWidth, PetalLength, PetalWidth]])
tree_model.predict([[6.2, 3.4, 5.4, 2.3]])
print("prediction:", tree_model.predict([[6.2, 3.4, 5.4, 2.3]]))

# Print Text Representation
export_graphviz(
    tree_model,
    out_file="iris_tree.dot",
    feature_names=['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm'],
    class_names=['Iris-setosa', 'Iris-versicolor', 'Iris-virginica' ],
    rounded=True,
    filled=True
)

# Plot the graph
clf = DecisionTreeClassifier(random_state=1234)
model = clf.fit(x, y)
fig = plt.figure(1)
_ = tree.plot_tree(clf,
                   feature_names=iris.feature_names,
                   class_names=iris.target_names,
                   filled=True)
fig.savefig("decistion_tree.png")
plt.show()
