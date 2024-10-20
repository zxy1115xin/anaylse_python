import numpy as np
from sklearn.decomposition import NMF
import scipy.io
import matplotlib.pyplot as plt

# 设置 NMF 模型
model = NMF(n_components=3, init='random', random_state=0,l1_ratio=0.1,alpha_H=0.0012,alpha_W=0.002)

# 读取 .mat 文件
mat_data = scipy.io.loadmat('.//data//moment_matrix.mat')
moment_out={}
# print("Keys in the MAT file:", mat_data.keys())
for num in range(8):
    data_array = mat_data[f'moment_{num+1}']
    data_array[data_array < 0] = 0

    # 进行矩阵分解，得到矩阵 W 和 H
    W = model.fit_transform(data_array )
    H = model.components_

    # 绘图
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
        f'Principal_{num+1}': W,
        f'Weight_{num+1}': H,
    }
    moment_out.update(data_dict)
    
scipy.io.savemat('multiple_data.mat',  moment_out)