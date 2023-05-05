'''
' @Author: Eureke
' @Date: 2023-03-15 13:25:08
' @LastEditTime: 2023-03-29 23:33:52
' @Description: 
'''
from PIL import Image
import numpy as np
import scipy.io.wavfile as wav
import pyofdm.codec
import pyofdm.nyquistmodem
import matplotlib.pyplot as plt

# pyofdm.codec.OFDM(nFreqSamples=64, 
# 	pilotIndices=[-21, -7, 7, 21],
# 	pilotAmplitude=1, 
# 	nData=12, 
# 	fracCyclic=0.25, 
# 	mQAM=2)

# Number of total frequency samples
totalFreqSamples = 2048
# Number of useful data carriers / frequency samples
sym_slots = 1512
# QAM Order
QAMorder = 2
# Total number of bytes per OFDM symbol
nbytes = sym_slots*QAMorder//8
# Distance of the evenly spaced pilots
distanceOfPilots = 12
pilotlist = pyofdm.codec.setpilotindex(nbytes, QAMorder, distanceOfPilots)

ofdm = pyofdm.codec.OFDM(pilotAmplitude = 16/9,
												nData=nbytes,
												pilotIndices = pilotlist,
												mQAM = QAMorder,
												nFreqSamples = totalFreqSamples)

# Take a uint8 as a simple example
row = np.random.randint(256,size=nbytes,dtype='uint8')
complex_signal = ofdm.encode(row)
print("complex_signal num:", complex_signal.size)

# plot OFDM symbol
plt.figure()
plt.title('OFDM Symbol')
plt.plot(complex_signal.real)
plt.plot(complex_signal.imag)
plt.show()

# plot OFDM complec spectrum
plt.figure()
plt.title("OFDM complex spectrum")
plt.xlabel("Normalised frequencies")
plt.ylabel("Frequency amplitudes")
plt.plot(np.abs(np.fft.fft(complex_signal[-totalFreqSamples:])/totalFreqSamples))
plt.show()

# open image and binary information
# img = Image.open("./Lab5/DC4_600x400.pgm")
img = Image.open("./Lab5/DC4_300x200.pgm")
# plot imag
plt.figure()
plt.title('Original Image')
plt.imshow(np.array(img),cmap="gray",vmin=0,vmax=255)
plt.show()

# binary data
tx_byte = np.array(img).ravel()

## add some random length dummy zero data to the start of the signal here
pad_num = nbytes - tx_byte.shape[0] % nbytes
tx_byte = np.pad(tx_byte, (0, pad_num), mode="constant", constant_values=127)
##

# OFDM encoding
complex_img_signal = np.array([ofdm.encode(tx_byte[i:i+nbytes]) 
	for i in range(0,tx_byte.size,nbytes)]).ravel()

# modulation
base_signal = pyofdm.nyquistmodem.mod(complex_img_signal)

## add some random length dummy zero to the start of the signal
random_pad_length = 50
base_signal = np.pad(base_signal, (random_pad_length, 0), mode="constant")
##

# save it as a wav file
wav.write('./Lab5/ofdm44100.wav',44100,base_signal)