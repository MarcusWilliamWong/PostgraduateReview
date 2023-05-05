'''
Author       : Eureke
Date         : 2023-03-08 20:28:50
LastEditors  : Marcus Wong
LastEditTime : 2023-03-08 21:16:57
Description  : 
'''
import numpy as np
import komm
from ImgInfo import imgInfo
from ModConfig import modConfig
from SimTrans import practiceBer

def concatenatedTransmit(img, inner_coder, outer_coder, method, orders, snr, base_amplitudes=1., phase_offset=0.):
  print("Start " + str(orders) + '-' + method + "modulation:")
  # inner&outer FEC encode
  BCHCoder = outer_coder
  imBin_copy = np.copy(img.imBin.reshape(int(img.imBin.size/BCHCoder.dimension), BCHCoder.dimension))
  # print('The shape after grouping: ', imBin_copy.shape)
  img.imBin_encoded = np.array([BCHCoder.encode(i) for i in imBin_copy]).ravel()
  ConvnCoder = inner_coder
  # create Convn encoder
  encoder = komm.ConvolutionalStreamEncoder(ConvnCoder, initial_state=0)
  imBin_copy = np.copy(img.imBin_encoded)
  img.imBin_encoded = encoder(imBin_copy)

  # initial modulation config
  mod_config = modConfig(method, orders, snr, base_amplitudes, phase_offset)
  mod_config.set_snr(10**(snr/10.))

  # modulated signal
  tx_data = mod_config.modulation.modulate(img.imBin_encoded)
  # add awgn
  rx_data = mod_config.awgn(tx_data)
  # demodulate at receiver
  rx_demod = mod_config.modulation.demodulate(rx_data)

  # decode demod signal
  tblen = 18
  decoder = komm.ConvolutionalStreamDecoder(ConvnCoder, traceback_length=tblen, input_type="hard")
  decoded_middle = decoder(np.append(rx_demod, np.zeros(2*tblen, dtype=np.int32)))
  rx_bin_inner = decoded_middle[tblen:]

  rx_bin_inner = rx_bin_inner.reshape(int(rx_bin_inner.size/BCHCoder.length), BCHCoder.length)
  rx_bin = np.array([BCHCoder.decode(i) for i in rx_bin_inner]).ravel()

  img.rx_bin = rx_bin

  ber = practiceBer(img.imBin, img.rx_bin)
  print('ber: ', ber)
  print('bit error ratio with BCH & Convn code: {:.3}%'.format(ber * 100))

  if (True):
    img.displayDemodImage()



if __name__ == '__main__':
  # open image and binary information
  fp = './Lab4/DC4_150x100.pgm'
  # fp = './Lab3/DC4_640x480.pgm'
  word_len = 8 # 256 bits per pixel
  img = imgInfo(fp, word_len)

  concatenatedTransmit(img=img, inner_coder=komm.ConvolutionalCode(feedforward_polynomials=[[0o7, 0o5]]), outer_coder=komm.BCHCode(mu=3, tau=1), method='psk', orders=4, snr=3., base_amplitudes=1., phase_offset=0.)
  concatenatedTransmit(img=img, inner_coder=komm.ConvolutionalCode(feedforward_polynomials=[[0o7, 0o5]]), outer_coder=komm.BCHCode(mu=3, tau=1), method='psk', orders=4, snr=0., base_amplitudes=1., phase_offset=0.)