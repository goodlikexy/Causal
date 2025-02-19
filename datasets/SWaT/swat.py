import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer

class SWaT:
    def __init__(self, options):
        self.options = options
        self.data_dict = {}
        self.seed = options['seed']
        self.num_vars = options['num_vars']
        self.data_dir = options['data_dir']
        self.window_size = options['window_size']
        self.shuffle = options['shuffle']

    def generate_example(self):
        # load data and save to csv
        try:
            df_label = pd.read_excel('datasets/SWaT/List_of_attacks_Final.xlsx', header=0, index_col=0)
            df_normal = pd.read_csv('datasets/SWaT/SWaT_Normal.csv', header=0, index_col=0)
            df_abnormal = pd.read_csv('datasets/SWaT/SWaT_Abnormal.csv', header=0, index_col=0)
        except:
            df_normal = pd.read_excel('datasets/SWaT/SWaT_Dataset_Normal_v1.xlsx', header=1)
            df_normal.to_csv('datasets/SWaT/SWaT_Normal.csv')
            df_abnormal = pd.read_excel('datasets/SWaT/SWaT_Dataset_Attack_v0.xlsx', header=1)
            df_abnormal.to_csv('datasets/SWaT/SWaT_Abnormal.csv')

        # clean label data
        df_label_clean = df_label.dropna(subset=['Start Time', 'End Time'], how='any')
        df_label_clean.drop(
            columns=['Start State', 'Attack', 'Expected Impact or attacker intent', 'Unexpected Outcome',
                     'Actual Change'], inplace=True)
        df_label_clean['Start Time'] = pd.to_datetime(df_label_clean['Start Time'])
        df_label_clean['Adjusted End Time'] = df_label_clean.apply(lambda row: pd.to_datetime(
            row['Start Time'].strftime('%Y-%m-%d') + ' ' + row['End Time'].strftime('%H:%M:%S')), axis=1)
        df_label_clean.to_csv('SWaT_label.csv')

        # clean normal data
        df_normal = df_normal.loc[df_normal['Normal/Attack'] == 'Normal']
        df_normal.drop(columns=[' Timestamp', 'Normal/Attack'], inplace=True)
        df_normal = df_normal[::10].reset_index(drop=True)

        # clean abnormal data
        df_abnormal.dropna(how='any', inplace=True)
        df_abnormal = df_abnormal[::].reset_index(drop=True)
        labels = np.zeros(df_abnormal.values[:, 1:-1].shape)
        df_abnormal['Adjusted Timestamp'] = pd.to_datetime(df_abnormal[' Timestamp'],
                                                           format=' %d/%m/%Y %I:%M:%S %p').dt.strftime(
            '%Y-%m-%d %H:%M:%S')
        df_abnormal['Adjusted Timestamp'] = pd.to_datetime(df_abnormal['Adjusted Timestamp'])

        # create dictionary for column names
        col_dic = {}
        for i in df_abnormal.columns.values[1:-2]:
            col_dic[i.lstrip()] = len(col_dic)

        test_x_lst = []
        test_label_lst = []
        for i in range(len(df_label_clean)):
            lower = df_label_clean.iloc[i]['Start Time']
            upper = df_label_clean.iloc[i]['Adjusted End Time']
            attack_lst = df_label_clean.iloc[i]['Attack Point'].split(",")
            attack_lst_ind = [col_dic[j.replace('-','').lstrip().upper()] for j in attack_lst]
            index_lst = np.array(df_abnormal.loc[(df_abnormal['Adjusted Timestamp'] >= lower) &
                                                 (df_abnormal['Adjusted Timestamp'] <= upper) &
                                                 (df_abnormal['Normal/Attack'] == 'Attack')].index.values)
            if len(index_lst) > 0:
                for j in attack_lst_ind:
                    labels[index_lst, j] = 1
                test_x_lst.append(
                    df_abnormal.iloc[min(index_lst) - 2 * 10 * self.window_size:min(index_lst) + 1 * 10 * self.window_size:10, 1:-2].values)
                test_label_lst.append(labels[min(index_lst) - 2 * 10 * self.window_size:min(index_lst) + 1 * 10 * self.window_size:10])
        x_n_list = []
        for i in range(0, len(df_normal), 1000):
            if i + 1000 < len(df_normal):
                x_n_list.append(df_normal.iloc[i:i + 1000].values)
        scaler = StandardScaler()
        scaler.fit(np.concatenate(x_n_list, axis=0))
        x_n_list = [scaler.transform(i) for i in x_n_list]
        test_x_lst = [scaler.transform(i) for i in test_x_lst]
        self.data_dict['x_n_list'] = np.array(x_n_list)
        if self.shuffle:
            np.random.seed(self.seed)
            indices = np.random.permutation(len(self.data_dict['x_n_list']))
            self.data_dict['x_n_list'] = self.data_dict['x_n_list'][indices]
        self.data_dict['x_ab_list'] = np.array(test_x_lst)
        self.data_dict['label_list'] = np.array(test_label_lst)

    def save_data(self):
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

        np.save(os.path.join(self.data_dir, 'x_n_list'), self.data_dict['x_n_list'])
        np.save(os.path.join(self.data_dir, 'x_ab_list'), self.data_dict['x_ab_list'])
        np.save(os.path.join(self.data_dir, 'label_list'), self.data_dict['label_list'])

    def load_data(self):
        self.data_dict['x_n_list'] = np.load(os.path.join(self.data_dir, 'x_n_list.npy'), allow_pickle=False)
        self.data_dict['x_ab_list'] = np.load(os.path.join(self.data_dir, 'x_ab_list.npy'), allow_pickle=True)
        self.data_dict['label_list'] = np.load(os.path.join(self.data_dir, 'label_list.npy'), allow_pickle=True)