import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np

class DataProcessor:
    def __init__(self, input_file, output_dir):
        self.input_file = input_file
        self.output_dir = output_dir
        self.df = pd.read_csv(input_file)

    def normalize_data(self, scaler_type='standard'):
        """归一化数据"""
        if scaler_type == 'minmax':
            scaler = MinMaxScaler()
        else:  # 默认使用 StandardScaler
            scaler = StandardScaler()
        
        columns_to_normalize = self.df.columns.difference(['Time'])
        self.df[columns_to_normalize] = scaler.fit_transform(self.df[columns_to_normalize])
        
        # 保存归一化后的数据
        output_file_path = f'{self.output_dir}/normalized_output_{scaler_type}.csv'
        self.df.to_csv(output_file_path, index=False)
        print(f"归一化后的数据已保存到 '{output_file_path}'")

    def remove_multihop_links(self, prefixes_to_remove):
        """去除多跳链路"""
        filtered_columns = [col for col in self.df.columns if not any(col.startswith(prefix) for prefix in prefixes_to_remove)]
        self.df = self.df[filtered_columns]

        # 保存处理后的数据
        output_file_path = f'{self.output_dir}/single_hop_data.csv'
        self.df.to_csv(output_file_path, index=False)
        print(f"去除多跳链路后的数据已保存到 '{output_file_path}'")

    def extract_anomaly_segments(self, label_file, min_count=15):
        """提取异常数据段"""
        label_b_df = pd.read_csv(label_file)
        row_num, col_num = label_b_df.shape
        def find_large_consecutive_ones(series, min_count):
            segments = []
            count = 0
            start_index = -1

            for i in range(len(series)):
                if series[i] == 1:
                    if count == 0:
                        start_index = i
                    count += 1
                else:
                    if count > min_count:
                        segments.append((start_index, i - 1))
                    count = 0
            
            if count > min_count:
                segments.append((start_index, len(series) - 1))
            return segments

        all_segments = []
        for col in label_b_df.columns:
            if col != 'Time':
                segments = find_large_consecutive_ones(label_b_df[col], min_count)  #这里label_b_df[col]是label_b_df的某一列
                all_segments.extend(segments)

        # 去除重叠或重复的段落
        non_overlapping_segments = []
        all_segments.sort()

        for start, end in all_segments:
            if not non_overlapping_segments:
                non_overlapping_segments.append((start, end))
            else:
                last_start, last_end = non_overlapping_segments[-1]
                if start <= last_end + 1:
                    non_overlapping_segments[-1] = (last_start, max(last_end, end))
                else:
                    non_overlapping_segments.append((start, end))

        # 保存每一段的最长提取数据
        for start, end in non_overlapping_segments:
            extracted_rows = self.df.iloc[start:end + 1]
            filename = f'{self.output_dir}/all_dantiao/anormaly_{start}_{end}.csv'
            extracted_rows.to_csv(filename, index=False)
            print(f"提取的最长行异常数据已保存到 '{filename}'")




    #过滤所有单跳指标中的正常指标
    def filter_normal_metric(self, segment_file, label_file, start, end):
        """
        根据标签文件中某段数据是否全为 0 来动态删除提取的异常段中的列
        """
        segment_df = pd.read_csv(segment_file)
        label_df = pd.read_csv(label_file)

        # 确保范围在合法行号内
        if start < 0 or end >= len(label_df):
            return f"指定的行范围 ({start}, {end}) 超出标签文件范围。"
        
        columns_to_remove = []
        for col in segment_df.columns:
            if col in label_df.columns and label_df[col][start:end + 1].eq(0).all():
                columns_to_remove.append(col)

        # 删除满足条件的列
        segment_df.drop(columns=columns_to_remove, inplace=True)

        # 保存过滤后的数据
        output_file_path = f'{self.output_dir}/all_anormaly_dantiao/filtered_{os.path.basename(segment_file)}'
        segment_df.to_csv(output_file_path, index=False)
        return f"根据标签文件过滤后的数据已保存到 '{output_file_path}'"
    
    #批量处理
    def batch_filter_normal_metric(self, anomaly_dir, label_file):
        """
        批量处理指定目录中的所有异常段文件。
        """
        label_df = pd.read_csv(label_file)
        anomaly_files = [f for f in os.listdir(anomaly_dir) if f.startswith('anormaly_') and f.endswith('.csv')]

        for file in anomaly_files:
            file_path = os.path.join(anomaly_dir, file)
            
            # 自动解析文件名中的范围信息，如 "anormaly_818_841.csv"
            try:
                start, end = map(int, os.path.splitext(file)[0].split('_')[1:])
            except ValueError:
                print(f"无法从文件名 '{file}' 中提取范围，跳过...")
                continue
            
            # 调用单文件过滤方法
            result = self.filter_normal_metric(file_path, label_file, start, end)
            print(result)

    def extract_normal_segments(self, label_file, min_count=50):
        """提取正常数据段：只有当所有列同时出现连续50个0时才算作一段
        Args:
            label_file: 标签文件路径
            min_count: 最小连续正常点数量，默认50
        """
        label_df = pd.read_csv(label_file)
        
        # 获取所有需要检查的列（排除Time列）
        check_columns = [col for col in label_df.columns if col != 'Time']
        
        # 找出所有列都为0的时间点
        all_normal = (label_df[check_columns] == 0).all(axis=1)
        
        segments = []
        count = 0
        start_index = -1

        # 寻找连续的正常段
        for i in range(len(all_normal)):
            if all_normal[i]:  # 所有列都为0
                if count == 0:
                    start_index = i
                count += 1
            else:  # 任一列不为0
                if count > min_count:  # 如果之前的连续正常长度大于阈值
                    segments.append((start_index, i - 1))
                count = 0  # 重置计数器
        
        # 处理最后一段
        if count > min_count:
            segments.append((start_index, len(all_normal) - 1))

        # 保存每一段正常数据
        normal_data_dir = f'{self.output_dir}/normal_segments'
        os.makedirs(normal_data_dir, exist_ok=True)
        
        for start, end in segments:
            extracted_rows = self.df.iloc[start:end + 1]
            filename = f'{normal_data_dir}/normal_{start}_{end}.csv'
            extracted_rows.to_csv(filename, index=False)
            print(f"提取的正常数据段已保存到 '{filename}'")

    def split_normal_segments(self, normal_segments_dir, window_size=50):
        """将正常数据段切分成固定长度的片段
        Args:
            normal_segments_dir: 存放正常数据段的目录
            window_size: 切分的窗口大小，默认50
        """
        # 创建输出目录
        split_data_dir = f'{self.output_dir}/normal_segments_split'
        os.makedirs(split_data_dir, exist_ok=True)
        
        # 获取所有正常数据段文件
        normal_files = [f for f in os.listdir(normal_segments_dir) if f.startswith('normal_') and f.endswith('.csv')]
        
        for file in normal_files:
            # 读取数据段
            file_path = os.path.join(normal_segments_dir, file)
            df = pd.read_csv(file_path)
            
            # 获取原始的起始位置
            start, end = map(int, file.replace('normal_', '').replace('.csv', '').split('_'))
            
            # 计算可以切分出多少个完整的窗口
            total_rows = len(df)
            num_windows = total_rows // window_size
            
            # 切分数据并保存
            for i in range(num_windows):
                window_start = i * window_size
                window_end = window_start + window_size
                window_df = df.iloc[window_start:window_end]
                
                # 计算实际的时间索引
                actual_start = start + window_start
                actual_end = start + window_end - 1
                
                # 保存切分后的数据
                output_file = f'{split_data_dir}/normal_split_{actual_start}_{actual_end}.csv'
                window_df.to_csv(output_file, index=False)
                print(f"切分的正常数据段已保存到 '{output_file}'")

    def merge_to_npy(self, split_segments_dir):
        """将所有切分好的CSV文件合并为一个NPY文件
        Args:
            split_segments_dir: 存放切分后数据段的目录
        """
        # 获取所有切分后的CSV文件
        csv_files = sorted([f for f in os.listdir(split_segments_dir) if f.startswith('normal_split_') and f.endswith('.csv')])
        
        if not csv_files:
            print("没有找到切分后的CSV文件")
            return
        
        # 读取第一个文件来获取维度信息
        first_df = pd.read_csv(os.path.join(split_segments_dir, csv_files[0]))
        window_size = len(first_df)  # 应该是50
        num_features = len(first_df.columns)
        num_samples = len(csv_files)
        
        # 创建numpy数组
        all_data = np.zeros((num_samples, window_size, num_features))
        
        # 读取所有文件并填充数组
        for i, file in enumerate(csv_files):
            df = pd.read_csv(os.path.join(split_segments_dir, file))
            all_data[i] = df.values
        
        # 保存为NPY文件
        output_file = f'{self.output_dir}/normal_segments_merged.npy'
        np.save(output_file, all_data)
        print(f"所有数据已合并保存到 '{output_file}'")
        print(f"数据形状: {all_data.shape}")


# 使用示例
input_file = '/home/hz/projects/AERCA/datasets/data_10_26/test_l/data/merged_output_normalized_2.csv'
output_dir = f'/home/hz/projects/AERCA/datasets/data_10_26/test_l/data_processed'
label_file = '/home/hz/projects/AERCA/datasets/data_10_26/10_26_anomaly_res/duodian/label_l_2.csv'






processor = DataProcessor(input_file, output_dir)
#processor.normalize_data(scaler_type='standard')  # 选择标准化
processor.remove_multihop_links(['Time', 'F2S1', 'F2F4', 'F2C1', 'C1S1', 'S1C1', 'C1F2', 'F4F2', 'S1F2'])
#提取异常数据段
# processor.extract_anomaly_segments(label_file, min_count=15)
#批量处理异常数据段
#processor.batch_anormal_metric(f'{output_dir}/all_dantiao',label_file)



#提取正常数据段
processor.extract_normal_segments(label_file, min_count=50)
#切分正常数据段
processor.split_normal_segments(f'{output_dir}/normal_segments', window_size=50)
processor.merge_to_npy(f'{output_dir}/normal_segments_split')
