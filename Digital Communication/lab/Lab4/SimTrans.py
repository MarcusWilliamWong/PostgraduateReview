'''
Author       : Eureke
Date         : 2023-03-08 15:06:27
LastEditors  : Marcus Wong
LastEditTime : 2023-03-08 20:47:30
Description  : 
'''
import numpy as np
from matplotlib import pyplot as plt
import komm
from ModConfig import modConfig

# input imag and FEC coder to encode
def encodeFEC(img, coder):
  if (isinstance(coder, komm._error_control_block.BCHCode)):
    BCHCoder = coder
    coder_type = "BCH"
    # there is a potential bug about (img.imBin.size/BCHCoder.dimension) if at the last code is not enough BCHCoder.dimension need to fill zero
    imBin_copy = np.copy(img.imBin.reshape(int(img.imBin.size/BCHCoder.dimension), BCHCoder.dimension))
    # print('The shape after grouping: ', imBin_copy.shape)
    img.imBin_encoded = np.array([BCHCoder.encode(i) for i in imBin_copy]).ravel()
    # print("The shape after BCH code: ", img.imBin_encoded.shape)
  elif (isinstance(coder, komm._error_control_convolutional.ConvolutionalCode)):
    ConvnCoder = coder
    coder_type = "Convn"
    # create Convn encoder
    encoder = komm.ConvolutionalStreamEncoder(ConvnCoder, initial_state=0)
    imBin_copy = np.copy(img.imBin)
    img.imBin_encoded = encoder(imBin_copy)

  return coder_type, img.imBin_encoded


def decodeFEC(rx_demod, coder):
  if (isinstance(coder, komm._error_control_block.BCHCode)):
    BCHCoder = coder
    coder_type = "BCH"
    # BCH code check and error recovery
    rx_demod = rx_demod.reshape(int(rx_demod.size/BCHCoder.length), BCHCoder.length)
    rx_bin = np.array([BCHCoder.decode(i) for i in rx_demod]).ravel()
  elif (isinstance(coder, komm._error_control_convolutional.ConvolutionalCode)):
    ConvnCoder = coder
    coder_type = "Convn"
    tblen = 18
    decoder = komm.ConvolutionalStreamDecoder(ConvnCoder, traceback_length=tblen, input_type="hard")
    # print(rx_demod.shape)
    # print(np.zeros(2*tblen, dtype=np.int32).shape)
    # print(type(rx_demod[0]))
    decoded_middle = decoder(np.append(rx_demod, np.zeros(2*tblen, dtype=np.int32)))
    rx_bin = decoded_middle[tblen:]
    
  return coder_type, rx_bin.astype(np.bool_)


# stimulate transmit single img with correction
def transmission(img, mod_config, coder):
  # transmission with FEC correction
  # modulated signal
  tx_data = mod_config.modulation.modulate(img.imBin_encoded)
  # add awgn
  rx_data = mod_config.awgn(tx_data)
  # demodulate at receiver
  rx_demod = mod_config.modulation.demodulate(rx_data)
  # decode using FEC decoder
  coder_type,  img.rx_bin = decodeFEC(rx_demod, coder)
  # compute ber with FEC
  ber = practiceBer(img.imBin, img.rx_bin)

  print('bit error ratio with {} code: {:.3}%'.format(coder_type, ber * 100))
  if (False):
    img.displayDemodImage()

  return ber


# stimulate transmit single img without correction
def transmissionNoCorrection(img, mod_config):
  # transmission with no correction
  tx_data = mod_config.modulation.modulate(img.imBin)
  rx_data = mod_config.awgn(tx_data)
  rx_bin = mod_config.modulation.demodulate(rx_data)
  ber = practiceBer(img.imBin, rx_bin)

  print('bit error ratio without FEC code: {:.3}%'.format(ber * 100))
  if (False):
    img.displayDemodImage()

  return ber


def repeatTransmit(img, coder, method, orders, snr_ctrl, base_amplitudes=1., phase_offset=0.):
  print("Start " + str(orders) + '-' + method + "modulation:")
  # use FEC to encode img
  coder_type, _ = encodeFEC(img, coder)

  # initial modulation config
  # snr from -3 to 9 dB
  mod_config = modConfig(method, orders, snr_ctrl[0], base_amplitudes, phase_offset)

  # save ber and snr of each trasmission single image 
  correction_ber_out = np.empty(0)
  nocorrection_ber_out = np.empty(0)
  snr_out = np.empty(0)
  for i in np.arange(snr_ctrl[0], snr_ctrl[1], snr_ctrl[2]):
    snr = 10**(i/10.)
    mod_config.set_snr(snr)
    correction_ber = transmission(img, mod_config, coder)
    nocorrection_ber = transmissionNoCorrection(img, mod_config)
  
    correction_ber_out = np.append(correction_ber_out, correction_ber)
    nocorrection_ber_out = np.append(nocorrection_ber_out, nocorrection_ber)
    snr_out = np.append(snr_out, i)
    print('snr(dB): ', i)
    # print('snr: ', mod_config.snr)

  print("Ber with correction: ", correction_ber_out)
  print("Ber without correction: ", nocorrection_ber_out)
  print("SNR: ", snr_out)

  if (True):
    plt.figure()
    plt.title(str(orders) + '-' + method.upper() + ' Snr(dB) vs Ber')
    if (coder_type == "BCH"):
      BCHCoder = coder
      plt.scatter(snr_out, correction_ber_out, color='r', label=('Ber with ' + '(' + str(BCHCoder.length) + ',' + str(BCHCoder.dimension) +')BCH'))
    elif (coder_type == "Convn"):
      ConvnCoder = coder
      plt.scatter(snr_out, correction_ber_out, color='r', label=('Ber with ' + '(' + str(ConvnCoder.num_output_bits) + ',' + str(ConvnCoder.num_input_bits) + ',' + str(ConvnCoder.overall_constraint_length + 1) +')Convolutional Code'))
    plt.plot(snr_out, nocorrection_ber_out, color='b', label='Ber without EFC')
    plt.yscale("log")
    plt.grid(True)
    plt.legend()
    plt.show()


# compute ber in practice
practiceBer = lambda tx_bin, rx_bin : np.sum([pix[0] != pix[1] for pix in zip(tx_bin, rx_bin)]) / tx_bin.size