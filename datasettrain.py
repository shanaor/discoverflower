import pathlib
import numpy as np
# import os
# import PIL
import PIL.Image
import tensorflow as tf
import tensorflow_datasets as tfds
import tarfile
import tempfile

# https://www.tensorflow.org/datasets/catalog/overview
dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
temp_dir = tempfile.mkdtemp()
archive = tf.keras.utils.get_file(
    origin=dataset_url,
    extract=False,
    cache_dir=temp_dir
)

with tarfile.open(archive, 'r:gz') as tar:
    tar.extractall(path='./media')