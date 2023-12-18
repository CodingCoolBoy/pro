import pickle
import numpy as np
from matplotlib import pyplot as plt
from scipy import io
Xd = pickle.load(open(r'dataset/RML2016.10a_dict.pkl', 'rb'), encoding='latin')
snrs, mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1, 0])
X = []
lbl = []
for mod in mods:
    for snr in snrs:
        X.append(Xd[(mod, snr)])
        for i in range(Xd[(mod, snr)].shape[0]):  lbl.append((mod, snr))

X = np.vstack(X)
# %%
np.random.seed(2016)  # 对预处理好的数据进行打包，制作成投入网络训练的格式，并进行one-hot编码
n_examples = X.shape[0]
n_train = n_examples * 0.7
train_idx = np.random.choice(range(0, n_examples), size=int(n_train), replace=False)
test_idx = list(set(range(0, n_examples)) - set(train_idx))  # label
X_train = X[train_idx]
X_test = X[test_idx]


def to_onehot(yy):
    yy1 = np.zeros([len(yy), max(yy) + 1])
    yy1[np.arange(len(yy)), yy] = 1
    return yy1


trainy = list(map(lambda x: mods.index(lbl[x][0]), train_idx))
Y_train = to_onehot(trainy)
Y_test = to_onehot(list(map(lambda x: mods.index(lbl[x][0]), test_idx)))

# for i in range(100):
#     n = int(np.where(Y_test[i] == 1)[0])
#     print(n)
#     print(Y_test[i])


i = 0
for signal in X_test:
    specg = plt.specgram(signal[0],NFFT=256,Fs=1,noverlap=128)
    n =int(np.where(Y_test[i] == 1)[0])
    plt.savefig('dataset/spec_test/moudle%d/spec%d.jpg'%(n ,i))

    i = i + 1
    if(i==100):
        break