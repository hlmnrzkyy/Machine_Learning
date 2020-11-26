import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import zipfile
import os
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing import image
from PIL import Image

# melakukan ekstraksi pada file zip
local_zip = 'messy-vs-clean-room.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/PycharmProjects/ML and ANN/tmp')
zip_ref.close()

base_dir = 'images'
train_dir = os.path.join(base_dir, '/PycharmProjects/ML and ANN/tmp/images/train')
validation_dir = os.path.join(base_dir, '/PycharmProjects/ML and ANN/tmp/images/val')

# membuka directory data latih
print("train image directory:")
print(os.listdir('/PycharmProjects/ML and ANN/tmp/images/train'))

# membuka direktori data validasi
print("validation image directory:")
print(os.listdir('/PycharmProjects/ML and ANN/tmp/images/val'))

# membuat direktori ruangan rapi pada direktori data training
train_clean_dir = os.path.join(train_dir, 'clean')

# membuat direktori ruangan berantakan pada direktori data training
train_messy_dir = os.path.join(train_dir, 'messy')

# membuat direktori ruangan rapi pada direktori data validasi
validation_clean_dir = os.path.join(validation_dir, 'clean')

# membuat direktori ruangan berantakan pada direktori data validasi
validation_messy_dir = os.path.join(validation_dir, 'messy')

train_datagen = ImageDataGenerator(
                    rescale=1./255,
                    rotation_range=20,
                    horizontal_flip=True,
                    shear_range=0.2,
                    fill_mode='nearest')

test_datagen = ImageDataGenerator(
                    rescale=1./255,
                    rotation_range=20,
                    horizontal_flip=True,
                    shear_range=0.2,
                    fill_mode='nearest')

train_generator = train_datagen.flow_from_directory(
        train_dir,  # direktori data latih
        target_size=(150, 150),  # mengubah resolusi seluruh gambar menjadi 150x150 piksel
        batch_size=4,
        # karena ini merupakan masalah klasifikasi 2 kelas maka menggunakan class_mode = 'binary'
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        validation_dir,  # direktori data validasi
        target_size=(150, 150),  # mengubah resolusi seluruh gambar menjadi 150x150 piksel
        batch_size=4,  # karena ini merupakan masalah klasifikasi 2 kelas maka menggunakan class_mode = 'binary'
        class_mode='binary')

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(512, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(
      train_generator,
      steps_per_epoch=25,  # berapa batch yang akan dieksekusi pada setiap epoch
      epochs=25,
      validation_data=validation_generator,  # menampilkan akurasi pengujian data validasi
      validation_steps=5,  # berapa batch yang akan dieksekusi pada setiap epoch
      verbose=2)

# Prediksi Gambar

filename = ["0.png", "1.png", "3.png",
            "4.png", "5.png", "6.png",
            "7.png", "8.png", "9.png"]

for i in range(0, len(filename)):
    plt.figure(1)
    plt.subplot(3, 3, i+1)
    img = Image.open(filename[i])
    img = img.resize((150, 150))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    images = np.vstack([x])
    classes = model.predict(images, batch_size=10)

    # Menampilkan keterangan pada gambar

    from PIL import ImageDraw

    if classes == 0:
        draw = ImageDraw.Draw(img)
        draw.text((10, 10), "CLEAN", (0, 0, 0))
    else:
        draw = ImageDraw.Draw(img)
        draw.text((10, 10), "MESSY", (0, 0, 0))
    plt.imshow(img)
plt.show()
