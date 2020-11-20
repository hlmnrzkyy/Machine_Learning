import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier

# ubah file csv menjadi dataframe
df = pd.read_csv('Mall_Customers.csv')

# tampilkan 3 baris pertama
print("-------------------------")
print("tampilkan 3 baris pertama")
print("-------------------------")
print(df.head(3))

# ubah nama kolom
df = df.rename(columns={'Gender': 'gender', 'Age': 'age',
                        'Annual Income (k$)': 'annual_income',
                        'Spending Score (1-100)': 'spending_score'})

# ubah data kategorik mmenjadi data numerik
df['gender'].replace(['Female', 'Male'], [0, 1], inplace=True)

# tampilkan data yang sudah di preprocess
print("-----------------------------")
print("tampilkan 3 baris pertama (1)")
print("-----------------------------")
print(df.head(3))

# menghilangkan kolom customer id dan gender
X = df.drop(['CustomerID', 'gender'], axis=1)

# membuat list yang berisi inertia
clusters = []
for i in range(1, 11):
    km = KMeans(n_clusters=i).fit(X)
    clusters.append(km.inertia_)

# tampilkan data yang sudah di preprocess
print("-----------------------------")
print("tampilkan 3 baris pertama (2)")
print("-----------------------------")
print(X.head(3))

# membuat plot inertia
fig, ax = plt.subplots(figsize=(8, 4))
sns.lineplot(x=list(range(1, 11)), y=clusters, ax=ax)
ax.set_title('Cari Elbow')
ax.set_xlabel('Clusters')
ax.set_ylabel('Inertia')

# membuat objek KMeans
km5 = KMeans(n_clusters=5).fit(X)

# menambahkan kolom label pada dataset
X['Labels'] = km5.labels_

# membuat plot KMeans dengan 5 klaster
plt.figure(figsize=(8,4))
sns.scatterplot(X['annual_income'], X['spending_score'], hue=X['Labels'],
                palette=sns.color_palette('hls', 5))
plt.title('KMeans dengan 5 Cluster')

# tampilkan data yang sudah di preprocess
print("-----------------------------")
print("tampilkan 3 baris pertama (3)")
print("-----------------------------")
print(X.head(3))
print(type(X['Labels']))

# Membuat program prediksi dengan K-NN
x1 = np.asarray([X['annual_income']])
print("x1", x1)
x2 = np.asarray([X['spending_score']])
print("x2", x2)
y1 = np.asarray([X['Labels']])
print("y1", y1)
X = []
Y = []
for i in range(0, len(x1[0,:])):
    print(x1[0, i], x2[0, i])
    X.append([x1[0, i], x2[0, i]])
    Y.append(y1[0, i])
print("X:\n", X)
print("Y:\n", Y)

neigh = KNeighborsClassifier(n_neighbors=5)
neigh.fit(X, Y) # need maintenance!

ulangi = "y"
while ulangi == "y":
    pemasukkan = input("Masukkan pemasukkan bulanan anda!")
    pemasukkan = float(pemasukkan)
    pengeluaran = input("Masukkan pengeluaran bulanan anda!")
    pengeluaran = float(pengeluaran)
    klaster = neigh.predict([[pemasukkan, pengeluaran]])
    print("Berdasarkan pemasukkan, dan pengeluaran anda,")
    print("anda masuk ke dalam klaster ke-", klaster[0])
    ulangi = input("prediksi lagi? (y/n)")
    ulangi.lower()

# Show the graph
plt.show()
