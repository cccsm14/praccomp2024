import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

## Skip downloand and explore dataset? or should I set 
##data_dir = "C:\Users\cccsm\praccomp2024\FinalProject\ImagesforModelTraining\images"
import pathlib
data_dir = "C:/Users/cccsm/.keras/datasets/flower_photos_extracted/flower_photos"
pathlib.Path(data_dir).with_suffix('')

batch_size = 32 ## Should we do 1 at a time?
img_height = 180
img_width = 180

train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)
class_names = train_ds.class_names
print(class_names)
