from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
import numpy as np
from matplotlib import pyplot as plt

# mendefinisikan atribut dan label pada dataset
x = np.asarray([[  1,   2,   3],
                [  2,   3,   1],
                [  3,   2,   1],
                [1.1, 1.9, 3.1],
                [2.1, 2.9, 0.9],
                [  3,   2,   1]])
y = np.asarray([0, 1, 2, 0, 1, 2])

# membuat model dengan decision tree classifier
clf = tree.DecisionTreeClassifier()

# mengevaluasi performa model dengan cross_val_score
scores = cross_val_score(clf, x, y, cv=2)
print("Cross Validation Score:", scores)

# membuat model Decision Tree
tree_model = DecisionTreeClassifier()

# melakukan pelatihan model terhadap data
tree_model.fit(x, y)

# membuat model Decision Tree
print(tree_model.predict([[2, 3.2, 0.8]]))

# Print Text Representation
export_graphviz(
    tree_model,
    out_file="my_tree.dot",
    feature_names=['feature_1', 'feature_2', 'feature_3'],
    class_names=['0', '1', '2' ],
    rounded=True,
    filled=True
)

# Plot the graph
clf = DecisionTreeClassifier(random_state=1234)
model = clf.fit(x, y)
fig = plt.figure(1)
_ = tree.plot_tree(clf,
                   feature_names=['feature_1', 'feature_2', 'feature_3'],
                   class_names=['0', '1', '2' ],
                   filled=True)
fig.savefig("my_tree.png")
plt.show()