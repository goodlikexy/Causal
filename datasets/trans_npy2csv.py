import numpy as np
import pandas as pd

# 加载npy文件
data = np.load('/home/hz/projects/AERCA/datasets/linear_point/x_n_list.npy')

# 将数据转换为DataFrame
df = pd.DataFrame(data[3])

# 将DataFrame保存为CSV文件

df.to_csv('/home/hz/projects/AERCA/datasets/linear_point/x_n_list[3].csv', index=False)
