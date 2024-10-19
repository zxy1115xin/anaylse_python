import numpy as np
from sklearn.decomposition import NMF
import scipy.io
import matplotlib.pyplot as plt

# 读取 .mat 文件
mat_data = scipy.io.loadmat('.//data//test.mat')
data_array = mat_data['angle_matrix_unit']
data_array[data_array < 0] = 0

# 设置 NMF 模型
n_components = 3  # 分解后的矩阵的潜在特征数
model = NMF(n_components=n_components, init='random', random_state=0, l1_ratio=0.16,alpha_W=0.0012)

# 进行矩阵分解，得到矩阵 W 和 H
W = model.fit_transform(data_array )
H = model.components_


plt.figure(figsize=(10, 6))
plt.plot(W[:, 0], label='Column 1', linewidth=2)
plt.plot(W[:, 1], label='Column 2', linewidth=2)
plt.plot(W[:, 2], label='Column 3', linewidth=2)
plt.title('Synergy')
plt.xlabel('Gait cycle (%)')
plt.ylabel('Y')
plt.grid(True)
plt.legend()
plt.show()

# 将这些数据保存到 .mat 文件中
data_dict = {
    'matrix': W,
    'vector': H,
}

scipy.io.savemat('multiple_data.mat', data_dict)

