import os
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import h5py
import cv2
import random
import glob
import math

from os import path
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras.callbacks import ModelCheckpoint
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.image import ResizeMethod
from random import shuffle
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import array_to_img
from keras import backend as K

AUTOTUNE = tf.data.AUTOTUNE
TRAIN_DIR = "./Data/"
bs = 4
scale = 4 # 3 or 4
patch_size = 96
img_size = int(patch_size * scale)

def loadImage(img, path_size, scale):
    img_size = int(path_size * scale)
    I = cv2.imread(img)
    I = random_crop(I, (img_size, img_size))
    y = I.copy()
    # Use different downsampling methods
    if np.random.randint(2): # x_scale sampling
        I = I[::scale, ::scale]
    else: #bilinear resizing
        I = cv2.resize(I, (path_size, path_size))
    return I, y

#flips a batch of images, flipMode is an integer in range(8)
def flip(x, flipMode):
    if flipMode in [4,5,6,7]:
        x = np.swapaxes(x,1,2)
    if flipMode in [1,3,5,7]:
        x = np.flip(x,1)
    if flipMode in [2,3,6,7]:
        x = np.flip(x,2)
    return x

def random_crop(image, crop_shape):
    if (crop_shape[0] < image.shape[1]) and (crop_shape[1] < image.shape[0]):
        x = random.randrange(image.shape[1]-crop_shape[0])
        y = random.randrange(image.shape[0]-crop_shape[1])
        
        return image[y:y+crop_shape[1], x:x+crop_shape[0], :]
    else:
        image = cv2.resize(image, crop_shape)
        return image

def test_edsr(model, x):
    
    p = model.predict(x)
    p = np.clip(p, 0, 255)
    plt.figure(figsize=(10,10))
    plt.subplot(121)
    plt.imshow(x.astype('uint8'))
    plt.title('Low Res')
    plt.subplot(122)
    plt.imshow(p[0].astype('uint8'))
    plt.title('Super Res')

def RMSE(diff): 
    return K.expand_dims(K.sqrt(K.mean(K.square(diff), [1,2,3])), 0)

def ssim(y_true, y_pred):
    return K.expand_dims(tf.image.ssim(y_true, y_pred, 255.), 0)

content_w=[0.1, 0.8, 0.1]
def content_fn(x):
    content_loss = 0
    n=len(content_w)
    for i in range(n): content_loss += RMSE(x[i]-x[i+n]) * content_w[i]
    return content_loss

def PSNR(super_resolution, high_resolution):
    """Compute the peak signal-to-noise ratio, measures quality of image."""
    # Max value of pixel is 255
    psnr_value = tf.image.psnr(high_resolution, super_resolution, max_val=255)[0]
    return psnr_value

def split_train_test(data, train_split=0.8, shuffle=True):
  if shuffle:
    np.random.shuffle(data)
  length = len(data)
  return data[:math.floor(length * train_split)], data[math.floor(length * train_split):]

def plot_imgs(imgs, img_labels, plt_shape, fig_size=(12, 12)):
  plt.figure(figsize=(12, 12))

  for i in range(len(imgs)):
      ax = plt.subplot(plt_shape[0], plt_shape[1], i + 1)
      plt.imshow(imgs[i])
      plt.title(f"{img_labels[i]} - {imgs[i].size} px")

def plot_metrics(_history, fig_size=(12, 12)):
  history = _history.history
  fig = plt.figure(figsize=fig_size)
  
  ax = plt.subplot(3,2, 1)
  ax.plot(history['loss'])
  ax.plot(history['val_loss'])
  plt.title('model loss')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(['train', 'val'], loc='upper left')

  ax = plt.subplot(3, 2, 2)
  ax.plot(history['acc'])
  ax.plot(history['val_acc'])
  plt.title('model accuracy')
  plt.ylabel('accuracy')
  plt.xlabel('epoch')
  plt.legend(['train', 'val'], loc='upper left')

  # summarize history for loss
  ax =  plt.subplot(3, 2, 3)
  ax.plot(history['ssim'])
  plt.title('model accuracy')
  plt.ylabel('ssim')
  plt.xlabel('epoch')
  plt.legend(['train', 'test'], loc='upper left')

  ax = plt.subplot(3,2, 4)
  ax.plot(history['PSNR'])
  plt.title('model accuracy')
  plt.ylabel('PSNR')
  plt.xlabel('epoch')
  plt.legend(['train', 'test'], loc='upper left')
  fig.tight_layout()
  plt.show()

def generate_images():
  """Generates additional images by transforming existing images
  """
  _generated_dir_path = os.path.join(TRAIN_DIR, "generated")

  if os.path.exists(_generated_dir_path) or len(glob.glob( _generated_dir_path+"**/*.png", recursive= True)) > 0:
    print("Generated directory exists with files. Not generating data.")
    return

  # get existing images from set5, set14
  IMAGES = sorted(glob.glob(TRAIN_DIR + '**/*.png', recursive=True))

  _existing_images = [Image.open(file_name) for file_name in IMAGES]

  _rotated_images = []
  for img in _existing_images:
    # Outputs random values from uniform distribution in between 0 to 4
    rn = tf.random.uniform(shape=(), maxval=4, dtype=tf.int32)
    # Here rn signifies number of times the image(s) are rotated by 90 degrees
    _roatetd_img = tf.image.rot90(np.asarray(img), rn)
    _rotated_images.append(_roatetd_img)

  _flipped_images = []

  # random flip the rotated images
  for img in _existing_images:
    _flipped_image = tf.image.flip_left_right(np.asarray(img))
    _flipped_images.append(_flipped_image)

  _generated_imgs = _flipped_images + _rotated_images

  count = 0
  print(f"Generated Images: {len(_generated_imgs)}")
  os.mkdir(_generated_dir_path)

  for _img in _generated_imgs:
    img = array_to_img(_img)
    img.save(_generated_dir_path+f"/t-{count}.png")
    count+=1
  
  return _generated_imgs

class ImageLoader(Sequence):
    
    #class creator, use generationMode = 'predict' for returning only images without labels
        #when using 'predict', pass only a list of files, not files and classes
    def __init__(self, files, path_size = 48, scale = 2, batchSize = 16, multi_loss=False, generationMode = 'train'):
        
        self.files = files
        self.batchSize = batchSize
        self.generationMode = generationMode
        self.path_size = path_size
        self.scale = scale
        self.multi_loss = multi_loss
        self
        assert generationMode in ['train', 'predict']
            

    #gets the number of batches this generator returns
    def __len__(self):
        l,rem = divmod(len(self.files), self.batchSize)
        return (l + (1 if rem > 0 else 0))
    
    #shuffles data on epoch end
    def on_epoch_end(self):
        if self.generationMode == 'train':
            shuffle(self.files)
        
    #gets a batch with index = i
    def __getitem__(self, i):
        
        #x are images   
        #y are labels
        
        images = self.files[i*self.batchSize:(i+1)*self.batchSize]
        
        x,y = zip(*[loadImage(f, self.path_size, self.scale) for f in images])
        
        x = np.stack(x, axis=0) # Low Res
        y = np.stack(y, axis=0) # High Res
        
        #cropping and flipping when training
        if self.generationMode == 'train':
                        
            flipMode = random.randint(0,7) #see flip functoin defined above
            x = flip(x, flipMode)
            y = flip(y, flipMode)
            
        if self.generationMode == 'predict':
            return x
        elif self.multi_loss:
            return [x, y], [y, np.zeros((len(x), 1))]
        else:
            return x, y