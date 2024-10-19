import numpy as np
from sklearn.decomposition import NMF

# 创建一个随机的非负矩阵作为输入数据
np.random.seed(42)
V = np.abs(np.random.randn(10, 5))  # 大小为 (10, 5) 的非负矩阵

# 设置 NMF 模型
n_components = 3  # 分解后的矩阵的潜在特征数
model = NMF(n_components=n_components, init='random', random_state=42, l1_ratio=0.5)

# 进行矩阵分解，得到矩阵 W 和 H
W = model.fit_transform(V)
H = model.components_

# 打印结果
print("Original Matrix (V):\n", V)
print("\nDecomposed Matrix W:\n", W)
print("\nDecomposed Matrix H:\n", H)
