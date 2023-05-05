'''
Author       : Eureke
Date         : 2023-03-06 14:29:36
LastEditors  : Marcus Wong
LastEditTime : 2023-03-08 17:33:57
Description  : 
'''
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

class imgInfo:
  def __init__(self, fp, word_len):
    self.imBin, self.imSize = self.openImagetoBin(fp)
    self.Npixels = self.imSize[1] * self.imSize[0]
    self.word_len = word_len
    self.imBin_encoded = None
    self.rx_bin = None

  # open image file
  def openImagetoBin(self, filePath):
    im = Image.open(filePath)
    if (True):
      plt.figure()
      plt.imshow(np.array(im),cmap="gray",vmin=0,vmax=255)
      plt.show()
    imBin = np.unpackbits(np.array(im))
    print('original shape: ', imBin.shape)
    return imBin, im.size

  # show demodulated image 
  def displayDemodImage(self):
    # demod signal with noise
    rx_im = np.packbits(self.rx_bin).reshape(self.imSize[1], self.imSize[0])
    plt.figure()
    plt.imshow(np.array(rx_im),cmap="gray",vmin=0,vmax=255)
    plt.show()