import os
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import pickle
import streamlit as st
import pickle

from os import path
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras.callbacks import ModelCheckpoint
from tensorflow.image import ResizeMethod
from random import shuffle
from keras import backend as K
from utils import *
from model import *
from tensorflow.keras.preprocessing.image import array_to_img

CHECKPOINT_DIR = "checkpoints/model"

def load_model(scale=2):
    _model = make_model(num_filters=64, num_of_residual_blocks=6)

    optim_edsr = keras.optimizers.Adam(
        learning_rate=keras.optimizers.schedules.PiecewiseConstantDecay(
            boundaries=[5000], values=[1e-4, 5e-5]
        )
    )

    _model.compile(optimizer=optim_edsr, loss="mae", metrics=[ssim, PSNR, "acc"])
    _model.load_weights(os.path.join(CHECKPOINT_DIR, f"sr_mymodel_loss_x{scale}.h5"))
    _session = K.get_session()

    return _model, _session

if __name__ == "__main__":
    _model, _session = load_model(4)