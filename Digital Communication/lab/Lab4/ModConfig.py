'''
Author       : Eureke
Date         : 2023-03-08 14:54:30
LastEditors  : Marcus Wong
LastEditTime : 2023-03-08 17:35:04
Description  : 
'''
import komm

class modConfig:
  def __init__(self, method, orders, snr, base_amplitudes, phase_offset):
    self.method = method
    self.orders = orders
    self.snr = snr
    self.base_amplitudes = base_amplitudes
    self.phase_offset = phase_offset
    self.modulation, self.awgn = self.set_modulation()

  # create komm's modulation object
  def set_modulation(self):
    if self.method == 'psk':
      modulation = komm.PSKModulation(self.orders, amplitude=self.base_amplitudes, phase_offset=self.phase_offset)
    elif self.method == 'qam':
      modulation = komm.QAModulation(self.orders, base_amplitudes=self.base_amplitudes, phase_offset=self.phase_offset)
    # Additive white gaussian noise(AWGN)
    awgn = komm.AWGNChannel(self.snr)
    return modulation, awgn

  # self-add snr
  def set_snr(self, new_snr):
    self.snr = new_snr
    self.modulation, self.awgn = self.set_modulation()