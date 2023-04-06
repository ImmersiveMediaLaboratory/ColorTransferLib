
import os
import glob
import json
import argparse
import torch
from utils.utils import calc_mean_score, save_json
from handlers.model_builder import Nima
from handlers.data_generator import TestDataGenerator
import tensorflow as tf
import torch.nn.functional as F

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)



def image_file_to_json(img_path):
    img_dir = os.path.dirname(img_path)
    img_id = os.path.basename(img_path).split('.')[0]

    return img_dir, [{'image_id': img_id}]


def image_dir_to_json(img_dir, img_type='jpg'):
    img_paths = glob.glob(os.path.join(img_dir, '*.'+img_type))

    samples = []
    for img_path in img_paths:
        img_id = os.path.basename(img_path).split('.')[0]
        samples.append({'image_id': img_id})

    return samples


def predict(model, img):
    out = tf.expand_dims(tf.convert_to_tensor(img), axis=0)
    return model.predict(out, workers=1, use_multiprocessing=True, verbose=1, steps=1)
