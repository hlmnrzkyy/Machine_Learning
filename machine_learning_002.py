from sklearn import datasets
from sklearn import tree
from sklearn.model_selection import cross_val_score

# Load iris dataset
iris = datasets.load_iris()
print(iris)

# mendefinisikan atribut dan label pada dataset
x = iris.data
y = iris.target

# membuat model dengan decision tree classifier
clf = tree.DecisionTreeClassifier()

# mengevaluasi performa model dengan cross_val_score
# clf/classifier: decision tree classifier
# x: attributes
# y: label
# cv: cross validation fold number
scores = cross_val_score(clf, x, y, cv=5)
print("Cross Validation Score:", scores)

