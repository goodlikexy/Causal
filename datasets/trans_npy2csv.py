import numpy as np
import pandas as pd

# 加载npy文件
data = np.load('/home/hz/projects/AERCA/datasets/data_10_26/test_d/data_processed/root_cause_timestamps.npy')
print("数据第一行")
#print(data[0][0])
# 将数据转换为DataFrame
df = pd.DataFrame(data)

# 将DataFrame保存为CSV文件

df.to_csv('/home/hz/projects/AERCA/datasets/data_10_26/test_d/data_processed/root_cause_timestamps.csv', index=False)
#print(df[0][0])