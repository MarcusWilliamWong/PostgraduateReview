'''
Author       : Eureke
Date         : 2023-03-08 16:30:21
LastEditors  : Marcus Wong
LastEditTime : 2023-03-08 23:33:38
Description  : 
'''
import numpy as np
import komm
from ImgInfo import imgInfo
from ModConfig import modConfig
from SimTrans import repeatTransmit



if __name__ == '__main__':
  # open image and binary information
  fp = './Lab4/DC4_150x100.pgm'
  # fp = './Lab3/DC4_640x480.pgm'
  word_len = 8 # 256 bits per pixel
  img = imgInfo(fp, word_len)

  '''
  print(img.imBin[:16])

  code = komm.ConvolutionalCode(feedforward_polynomials=[[0o7, 0o5]])
  encoder = komm.ConvolutionalStreamEncoder(code, initial_state=0)

  new_m = encoder(img.imBin)
  print(new_m[:32])

  decoder = komm.ConvolutionalStreamDecoder(code, traceback_length=4, input_type="hard")
  
  decoded_m_final = decoder(np.append(new_m[:32], np.zeros(8, dtype=np.int32)))
  # decoded_m_final = decoder(np.zeros(2*8, dtype=np.int32))
  print(decoded_m_final[4:])
  print(decoded_m_final.shape)
  '''

  snr_ctrl = [-3., 9.,  0.5]
  snr_ctrl = [3., 3.5,  0.5]
  snr_ctrl = [0., 0.5,  0.5]
  # qpsk modulation with convn code
  repeatTransmit(img=img, coder=komm.ConvolutionalCode(feedforward_polynomials=[[0o7, 0o5]]), method='psk', orders=4, snr_ctrl=snr_ctrl, base_amplitudes=1., phase_offset=0.)