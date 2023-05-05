'''
Author       : Eureke
Date         : 2023-03-08 17:42:10
LastEditors  : Marcus Wong
LastEditTime : 2023-03-08 17:42:40
Description  : 
'''
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import komm
from scipy import special
tx_im = Image.open('./Lab4/DC4_150x100.pgm')
Npixels = tx_im.size[1]*tx_im.size[0]
# print(np.shape(tx_im))
# print(Npixels)
plt.figure()
plt.imshow(np.array(tx_im),cmap="gray",vmin=0,vmax=255)
plt.show()
tx_bin = np.unpackbits(np.array(tx_im))
print(tx_bin.shape)
print(tx_bin.dtype)
print(tx_bin)

def slice_Array(arrayToBeSliced,k):
    x_2d = arrayToBeSliced.reshape(-1, k)  # 将一维数组重塑为二维数组，每行有4个元素
    return x_2d

def show_Pic(array):
    rx_im = np.packbits(array).reshape(tx_im.size[1],tx_im.size[0])
    plt.imshow(np.array(rx_im),cmap="gray",vmin=0,vmax=255)
    plt.show()
    return 0

def cov(message):
    print(message.size)

    code = komm.ConvolutionalCode(feedforward_polynomials=[[0o7, 0o5]])
    tblen = 18
    encoder = komm.ConvolutionalStreamEncoder(code, initial_state=0)
    decoder = komm.ConvolutionalStreamDecoder(code, traceback_length=message.size, input_type="hard")

    encoded_m = encoder(message)
#     print('encoded: :', encoded_m)
    decoded_m = decoder(encoded_m)
    decoded_m_final = decoder(np.zeros(2*message.size, dtype=np.int32))
#     print(decoded_m)
#     print(decoded_m_final)
    return decoded_m_final

arr = np.zeros((int(120000/8), 8), dtype=np.int32)
message = slice_Array(tx_bin,8)
print(message)
for i in range(int(120000/8)):
    arr[i] = cov(message[i])
arr_1d = arr.flatten()

show_Pic(arr_1d)