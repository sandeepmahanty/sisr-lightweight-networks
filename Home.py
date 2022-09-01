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


@st.cache(allow_output_mutation=True)
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


@st.cache(allow_output_mutation=True)
def load_training_history(scale=2):
    _history = {}

    with open(f"train_history_dict_x{scale}") as hist_file:
        _history = pickle.load(hist_file)

    return _history


def resize_image(_img):
    return tf.image.resize(np.asarray(_img), (96, 96), ResizeMethod.BICUBIC)


def set_page_title():
    st.markdown("<h1 style='text-align: center;'>Single Image Super Resolution</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center;'>With</h2>", unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center;'>Lightweight Neural Networks</h1>", unsafe_allow_html=True)


def set_page_header():
    st.subheader("What is SISR ?")
    st.write(
        """SR or Super-Resolution is a task of computer vision applications where a HR (High Resolution)
     image is generated from mutiple LR (Low Resolution) images. SISR or Single Image Super Resolution is a subtask of SR
     where a single LR image is used to generate a HR image."""
    )
    st.subheader("What are lightweight networks ?")
    st.write(
        """Lightweight networks are those networks which have very less number of trainable parameters.
     These networks are not very deep and hence have a small memory footprint and require considerably less cpu compute.
     Although there is no fixed threshold for how many parameters a lightweight network should have, but generally any
     network having less than 500K-600K params can be considered lightweight."""
    )


if __name__ == "__main__":

    _selected_scale = st.selectbox(
        "How much would you like to upscale your image ?", ("4X", "2X")
    )

    st.write(f"Selected Option: {_selected_scale}")
    with st.spinner("Loading the model, please wait ..."):
        # load model with saved weights and keras backend as session
        _scale_factor = 4 if _selected_scale == "4X" else 2
        print(f"Scale: {_scale_factor}")
        _model, _session = load_model(2)

    st.success("Model successfully loaded!")

    # K.set_session(_session)

    set_page_title()
    set_page_header()

    # setup file upload
    _img_file = st.file_uploader("Choose an image to scale up", type=["png", "jpg"])

    # if image file exists and has content
    if _img_file is not None:
        try:
            _img = Image.open(_img_file)

            with st.spinner("Please wait while your image is being upscaled ..."):
                _lr_img = resize_image(_img)

                st.markdown("### Chosen image")
                st.image(array_to_img(_lr_img), caption=_img_file.name)

                # predict the upscaled image
                _pred_img = _model.predict_step(_lr_img)

                print(_pred_img.shape)

                if _pred_img is not None:
                    st.markdown("### Upscaled image")
                    st.image(
                        np.asarray(_pred_img),
                        caption=f"{_img_file.name} upscaled by {_selected_scale}",
                    )
                    st.balloons()
                else:
                    st.error("Failed to upscale the image due to an error.")
        except:
            st.error("Failed to upscale the image due to an error.")
