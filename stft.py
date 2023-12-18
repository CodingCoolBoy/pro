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
real_part = rml_data[('PAM4', -10)][13][1]
imag_part = rml_data[('PAM4', -10)][13][0]
# 进行IQ解调
complex_signal = real_part + 1j * imag_part

# 进行STFT
frequencies, times, Zxx = stft(real_part)

# 绘制时频图
plt.figure(figsize=(10, 6))
plt.imshow(np.abs(Zxx), aspect='auto', extent=[times.min(), times.max(), frequencies.min(), frequencies.max()], origin='lower')

# 设置图形标题和轴标签
plt.title('时频图')
plt.xlabel('时间')
plt.ylabel('频率 (Hz)')

# 显示颜色条
plt.colorbar(label='振幅')

# 显示图形
plt.show()

