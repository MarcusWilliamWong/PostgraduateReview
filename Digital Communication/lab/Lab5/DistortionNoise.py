'''
' @Author: Eureke
' @Date: 2023-03-26 11:26:49
' @LastEditTime: 2023-03-30 01:09:54
' @Description: 
'''
from PIL import Image
import numpy as np
import scipy.io.wavfile as wav
import pyofdm.codec
import pyofdm.nyquistmodem
import matplotlib.pyplot as plt

# Number of total frequency samples
totalFreqSamples = 2048

# Number of useful data carriers / frequency samples
sym_slots = 1512

# QAM Order
QAMorder = 2

# Total number of bytes per OFDM symbol
nbytes = sym_slots * QAMorder // 8

# Distance of the evenly spaced pilots
distanceOfPilots = 12
pilotlist = pyofdm.codec.setpilotindex(nbytes, QAMorder, distanceOfPilots)

ofdm = pyofdm.codec.OFDM(pilotAmplitude=16/9, 
                        nData=nbytes, 
                        pilotIndices=pilotlist,
                        mQAM=QAMorder,
                        nFreqSamples=totalFreqSamples)

samp_rate, base_signal = wav.read("./Lab5/ofdm44100_reverb.wav")

# append some extra zeros to the base_signal
extra_pad_length = 60
base_signal = np.pad(base_signal, (0, extra_pad_length), "constant")

complex_signal = pyofdm.nyquistmodem.demod(base_signal)

# find the start of the OFDM symbol
searchRangeForPilotPeak = 8
cc, sumofimag, offset = ofdm.findSymbolStartIndex(complex_signal, searchrangefine=searchRangeForPilotPeak)
print("Symbol start sample index =", offset)

Nsig_sym = 159
ofdm.initDecode(complex_signal, 25)
rx_byte = np.uint8([ofdm.decode()[0] for i in range(Nsig_sym)]).ravel()
rx_byte = 255 - rx_byte

rx_byte = rx_byte[:60000].reshape(200, 300)
receive_img = Image.fromarray(rx_byte)
plt.imshow(receive_img, plt.cm.gray)

# calculate bit error ratio
# origin_img = Image.open("./Lab5/DC4_600x400.pgm")
origin_img = Image.open("./Lab5/DC4_300x200.pgm") 
origin_img = np.array(origin_img)
# ber = np.sum(origin_img != receive_img) / origin_img.size
# compute ber in practice
practiceBer = lambda tx_bin, rx_bin : np.sum([pix[0] != pix[1] for pix in zip(tx_bin, rx_bin)]) / tx_bin.size
ber = practiceBer(origin_img, rx_byte)
print("Bit error ratio : ", ber)



def receive(wave_file):
  samp_rate, base_signal = wav.read(wave_file)
  # append some extra zeros to the base_signal
  extra_pad_length = 60
  base_signal = np.pad(base_signal, (0, extra_pad_length), "constant")
  complex_signal = pyofdm.nyquistmodem.demod(base_signal)
    
  Nsig_sym = 159
  ofdm.initDecode(complex_signal, 25)
  rx_byte = np.uint8([ofdm.decode()[0] for i in range(Nsig_sym)]).ravel()
  rx_byte = 255 - rx_byte
    
  rx_byte = rx_byte[:60000].reshape(200, 300)
  receive_img = Image.fromarray(rx_byte)
  plt.imshow(np.array(receive_img),cmap="gray",vmin=0,vmax=255)
  plt.show()
  # plt.imshow(receive_img, plt.cm.gray)
    
  # calculate bit error ratio
  origin_img = Image.open("./Lab5/DC4_300x200.pgm")
  origin_img = np.array(origin_img)
  ber = practiceBer(origin_img, rx_byte)
  print("Bit error ratio = ", ber)


receive("./Lab5/ofdm44100_reverb.wav")

# receive("./Lab5/ofdm44100_reverb100.wav")

# receive("./Lab5/ofdm44100_reverb70.wav")

# receive("./Lab5/ofdm44100_reverb50.wav")

# receive("./Lab5/ofdm44100_reverb30.wav")

# receive("./Lab5/ofdm44100_reverb0.wav")

# receive("./Lab5/ofdm44100_damping100.wav")

# receive("./Lab5/ofdm44100_damping70.wav")

# receive("./Lab5/ofdm44100_damping50.wav")

# receive("./Lab5/ofdm44100_damping30.wav")

# receive("./Lab5/ofdm44100_damping0.wav")

# receive("./Lab5/ofdm44100_.wav")

receive("./Lab5/ofdm44100_noise0.1.wav")

receive("./Lab5/ofdm44100_noise0.05.wav")

receive("./Lab5/ofdm44100_noise0.01.wav")