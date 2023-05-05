'''
Author       : Eureke
Date         : 2023-03-08 15:00:18
LastEditors  : Marcus Wong
LastEditTime : 2023-03-08 19:39:56
Description  : 
'''

import komm
import numpy as np

message = np.array([1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1])
# message = np.array([1, 0, 1, 1, 1, 0, 1, 1, 0, 0])
print(message.size)

# code = komm.ConvolutionalCode(feedforward_polynomials=[[0o5, 0o3, 0o3, 0o4]])
code = komm.ConvolutionalCode(feedforward_polynomials=[[0o7, 0o5]])
print(type(code))
# print(type(komm.BCHCode(mu=3, tau=1)))
# code = komm.ConvolutionalCode(feedforward_polynomials=[[0o1]])
print('output, input, length', code.num_output_bits, code.num_input_bits, code.overall_constraint_length)
print('order:', code.memory_order)
tblen = 18
encoder = komm.ConvolutionalStreamEncoder(code, initial_state=0)
decoder = komm.ConvolutionalStreamDecoder(code, traceback_length=message.size, input_type="hard")

# for i in range(2):
encoded_m = encoder(message)
print('encoded: :', encoded_m)
decoded_m = decoder(encoded_m)
decoded_m_final = decoder(np.zeros(2*message.size, dtype=np.int32))
print(decoded_m)
print(decoded_m_final)

# rx_enc = psk.demodluate(rx_data, decision_method="hard")

# convolutional_code = komm.ConvolutionalCode([[0o7, 0o5]])
# convolutional_decoder = komm.ConvolutionalStreamDecoder(convolutional_code, traceback_length=10)
# convolutional_decoder([1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1])
# array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
# convolutional_decoder(np.zeros(2*10, dtype=np.int))
# array([1, 0, 1, 1, 1, 0, 1, 1, 0, 0])