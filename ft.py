import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft
# 假设已有IQ信号 complex_signal
with open(r'dataset/RML2016.10a_dict.pkl', 'rb') as p_f:
    rml_data = pickle.load(p_f, encoding="latin-1")
all_modulation_types = list(rml_data.keys())
# print("Available Modulation Types:", all_modulation_types)
selected_modulation_type = all_modulation_types[0]
print(all_modulation_types)
# # 将实部和虚部分离
real_part = rml_data[('PAM4', -10)][1][1]
imag_part = rml_data[('PAM4', -10)][1][0]
# 进行IQ解调
iq_signal = real_part + 1j * imag_part
# 进行FFT
fft_result = np.fft.fft(real_part)
# 计算频率轴
sampling_rate = 1000  # 采样率
freq_axis = np.fft.fftfreq(len(iq_signal), d=1/sampling_rate)
# 绘制频谱
plt.figure(figsize=(10, 6))
plt.plot(freq_axis, np.abs(fft_result))
plt.title('IQ信号的频谱')
plt.xlabel('频率 (Hz)')
plt.ylabel('振幅')
plt.grid(True)
plt.show()
