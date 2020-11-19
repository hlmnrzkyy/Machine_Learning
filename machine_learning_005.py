import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# buat data jumlah kamar
bedrooms = np.array([1, 1, 2, 2, 3, 4, 4, 5, 5, 5])

# data harga rumah. asumsi dalam dollar
house_price = np.array([15000, 18000, 27000, 34000, 50000, 68000, 65000, 81000, 85000, 90000])

# menampilkan scatter plot dari dataset
plt.scatter(bedrooms, house_price)

# latih model dengan Linear Regression.fit()
bedrooms = bedrooms.reshape(-1, 1)
linreg = LinearRegression()
linreg.fit(bedrooms, house_price)

# menampilkan plot hubungan antara jumlah kamar dengan harga rumah
plt.scatter(bedrooms, house_price)
plt.plot(bedrooms, linreg.predict(bedrooms))

# predict with linear regression
reg = LinearRegression().fit(bedrooms, house_price)
jumlah_kamar = input("jumlah kamar tidur yang diinginkan: ")
jumlah_kamar = int(jumlah_kamar)
harga_kamar = reg.predict(np.array([[jumlah_kamar]]))
print("harga kamar dengan", jumlah_kamar, "kamar tidur yang diinginkan adalah: $", harga_kamar[0])
print(reg.predict(np.array([[jumlah_kamar]])))

# show the plot
plt.show()
