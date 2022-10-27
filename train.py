# from: https://colab.research.google.com/github/khanhlvg/tflite_raspberry_pi/blob/main/object_detection/Train_custom_model_tutorial.ipynb#scrollTo=Jbl8z9_wBPlr

import numpy as np
import os

from tflite_model_maker.config import ExportFormat, QuantizationConfig
from tflite_model_maker import model_spec
from tflite_model_maker import object_detector

from tflite_support import metadata

import tensorflow as tf
assert tf.__version__.startswith('2')

tf.get_logger().setLevel('ERROR')
from absl import logging
logging.set_verbosity(logging.ERROR)


train_data = object_detector.DataLoader.from_pascal_voc(
    '/home/nathan/Desktop/lab/beeVideos_use/queenBee_train',
    '/home/nathan/Desktop/lab/beeVideos_use/queenBee_train',
    ['queenBee']
)

val_data = object_detector.DataLoader.from_pascal_voc(
    '/home/nathan/Desktop/lab/beeVideos_use/queenBee_validate',
    '/home/nathan/Desktop/lab/beeVideos_use/queenBee_validate',
    ['queenBee']
)

spec = model_spec.get('efficientdet_lite0')

model = object_detector.create(train_data, model_spec=spec, batch_size=4, train_whole_model=True, epochs=20, validation_data=val_data)

model.evaluate(val_data)

model.export(export_dir='.', tflite_filename='node_redo_oct5.tflite')

model.evaluate_tflite('node_redo_oct5.tflite', val_data)