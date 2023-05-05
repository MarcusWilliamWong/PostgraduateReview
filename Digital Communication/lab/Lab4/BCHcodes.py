'''
Author       : Eureke
Date         : 2023-03-05 09:22:37
LastEditors  : Marcus Wong
LastEditTime : 2023-03-08 21:19:49
Description  : 
'''
import numpy as np
import komm
from ImgInfo import imgInfo
from ModConfig import modConfig
from SimTrans import repeatTransmit

if __name__ == "__main__":
  # open image and binary information
  fp = './Lab4/DC4_150x100.pgm'
  # fp = './Lab3/DC4_640x480.pgm'
  word_len = 8 # 256 bits per pixel
  img = imgInfo(fp, word_len)
  # BCH code
  # Length = 2^miu - 1
  # message length = tau = 1
  # code = komm.BCHCode(mu=3, tau=1)
  # n, k = code.length, code.dimension
  # print(code.generator_polynomial)
  # print(code.generator_matrix)

  # message = np.array([1, 0, 0, 1])
  # recvword = code.encode(message)
  # print(recvword)
  # message_decoded = code.decode(recvword)
  # print(message_decoded)
  
  snr_ctrl = [-3., 9.,  0.5]
  snr_ctrl = [3., 3.5,  0.5]
  snr_ctrl = [0., 0.5,  0.5]
  # qpsk modulation with BCH code
  repeatTransmit(img=img, coder=komm.BCHCode(mu=3, tau=1), method='psk', orders=4, snr_ctrl=snr_ctrl, base_amplitudes=1., phase_offset=0.)
  repeatTransmit(img=img, coder=komm.BCHCode(mu=4, tau=3), method='psk', orders=4, snr_ctrl=snr_ctrl, base_amplitudes=1., phase_offset=0.)
  repeatTransmit(img=img, coder=komm.BCHCode(mu=5, tau=7), method='psk', orders=4, snr_ctrl=snr_ctrl, base_amplitudes=1., phase_offset=0.)
  repeatTransmit(img=img, coder=komm.BCHCode(mu=6, tau=13), method='psk', orders=4, snr_ctrl=snr_ctrl, base_amplitudes=1., phase_offset=0.)