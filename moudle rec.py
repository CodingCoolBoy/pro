import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft
with open(r'dataset/RML2016.10a_dict.pkl', 'rb') as p_f:
     rml_data = pickle.load(p_f, encoding="latin-1")
all_modulation_types = list(rml_data.keys())
# print("Available Modulation Types:", all_modulation_types)
sample_data = rml_data[('PAM4', -10)][1]
time_axis = np.arange(sample_data.shape[1])
real_part = rml_data[('PAM4', -10)][1][1]
imag_part = rml_data[('PAM4', -10)][1][0]
# 进行IQ解调
complex_signal = real_part + 1j * imag_part
# 绘制波形图
# plt.plot(time_axis, complex_signal[:], label='I ')
# plt.plot(time_axis, sample_data[1, :], label='REAL ')

# 设置图例和标题
# plt.title('IQ Signal Waveform')
# plt.xlabel('Time')
# plt.ylabel('Amplitude')
# plt.legend()
# plt.show()

plt.specgram(real_part,NFFT=256,Fs=1,noverlap=128)
plt.show()



