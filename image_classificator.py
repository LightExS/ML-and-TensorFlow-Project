import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import layers
from tensorflow.python.keras.models import Sequential
import pathlib
import matplotlib.pyplot as plt
from PIL import Image


class ImageClassificator:
    def __init__(self, img_height, img_width, batch_size):
        self.img_height = img_height
        self.img_width = img_width
        self.batch_size = batch_size

    def load_dataset(self, dataset_url=None, dataset_name=None, validation_spit=0.2):
        data_dir = tf.keras.utils.get_file(dataset_name, origin=dataset_url, untar=True)
        self.data_dir = pathlib.Path(data_dir)

        self.train_ds = tf.keras.utils.image_dataset_from_directory(
            self.data_dir,
            validation_split=validation_spit,
            subset="training",
            seed=123,
            image_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
        )

        self.val_ds = tf.keras.utils.image_dataset_from_directory(
            self.data_dir,
            validation_split=validation_spit,
            subset="validation",
            seed=123,
            image_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
        )
        self.class_names = self.train_ds.class_names

    def show_accuracy_graph(self):
        acc = self.history.history["accuracy"]

        val_acc = self.history.history["val_accuracy"]

        loss = self.history.history["loss"]
        val_loss = self.history.history["val_loss"]

        epochs_range = range(self.epochs)

        plt.figure(figsize=(8, 8))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label="Training Accuracy")
        plt.plot(epochs_range, val_acc, label="Validation Accuracy")
        plt.legend(loc="lower right")
        plt.title("Training and Validation Accuracy")

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label="Training Loss")
        plt.plot(epochs_range, val_loss, label="Validation Loss")
        plt.legend(loc="upper right")
        plt.title("Training and Validation Loss")
        plt.show()

    def train_model(self, epochs, optimizer="adam"):
        self.epochs = epochs
        self.model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )
        self.history = self.model.fit(
            self.train_ds, validation_data=self.val_ds, epochs=epochs
        )

    def save_model(self, path):
        self.model.save(path)

    def cache_dataset(self):
        AUTOTUNE = tf.data.AUTOTUNE
        self.train_ds = (
            self.train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
        )
        self.val_ds = self.val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    def add_data_augmentation(self, random_rotation=0.1, random_zoom=0.1):
        self.data_augmentation = tf.keras.Sequential(
            [
                tf.keras.layers.RandomFlip(
                    "horizontal", input_shape=(self.img_height, self.img_width, 3)
                ),
                tf.keras.layers.RandomRotation(random_rotation),
                tf.keras.layers.RandomZoom(random_zoom),
            ]
        )

    def create_model(self, dropout_percent=0.2):
        num_classes = len(self.class_names)
        self.model = Sequential(
            [
                self.data_augmentation,
                tf.keras.layers.Rescaling(1.0 / 255),
                layers.Conv2D(16, 3, padding="same", activation="relu"),
                layers.MaxPooling2D(),
                layers.Conv2D(32, 3, padding="same", activation="relu"),
                layers.MaxPooling2D(),
                layers.Conv2D(64, 3, padding="same", activation="relu"),
                layers.MaxPooling2D(),
                layers.Dropout(dropout_percent),
                layers.Flatten(),
                layers.Dense(128, activation="relu"),
                layers.Dense(num_classes, name="outputs"),
            ]
        )

    def load_model(self, path):
        self.model = tf.keras.models.load_model(path)

    def predict(self, img_array):
        predictions = self.model.predict(img_array)
        score = tf.nn.softmax(predictions[0])

        print(
            "This image most likely belongs to {} with a {:.2f} percent confidence.".format(
                self.class_names[np.argmax(score)], 100 * np.max(score)
            )
        )


def create_model():
    img_height = 180
    img_width = 180
    batch_size = 32

    model = ImageClassificator(img_height, img_width, batch_size)
    model.load_dataset(
        "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz",
        "flower_photos",
    )

    model.cache_dataset()
    model.add_data_augmentation()
    model.create_model()
    model.train_model(50)
    model.save_model("saved_model/model_50_epoch")

    image_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/592px-Red_sunflower.jpg"
    image_path = tf.keras.utils.get_file("Red_sunflower", origin=image_url)

    img = tf.keras.utils.load_img(image_path, target_size=(img_height, img_width))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    model.predict(img_array)
    model.show_accuracy_graph()


def load_model():
    img_height = 180
    img_width = 180
    batch_size = 32
    model = ImageClassificator(img_height, img_width, batch_size)
    model.load_model("saved_model/model_50_epoch")
    model.load_dataset(
        "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz",
        "flower_photos",
    )

    image_url = "https://images.squarespace-cdn.com/content/v1/56bf55504c2f85a60a9b9fe5/1635897793784-OD3181KEQJ2AV5QTEEK6/SunflowerSunset.jpg?format=1000w"
    image_path = tf.keras.utils.get_file(
        "5", origin=image_url
    )  # if you changing photo url, change name of the picture as well, because it will try to open older picture

    img = tf.keras.utils.load_img(image_path, target_size=(img_height, img_width))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    model.predict(img_array)


if __name__ == "__main__":
    # create_model()
    load_model()
