import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer

class MSDS:
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
        df_label = pd.read_csv('datasets/MSDS/labels.csv')
        df_normal = pd.read_csv('datasets/MSDS/train.csv')
        df_abnormal = pd.read_csv('datasets/MSDS/test.csv')

        df_normal, df_abnormal = df_normal.values[::5, 1:], df_abnormal.values[::5, 1:]
        df_label = df_label.values[::5, 1:]
        labels = np.max(df_label, axis=1)

        x_n_list = []
        for i in range(0, len(df_normal), 10000):
            if i + 10000 < len(df_normal):
                x_n_list.append(df_normal[i:i + 10000])
        test_x_lst = []
        label_lst = []

        for i in np.where(labels == 1)[0]:
            if i - 2 * self.window_size > 0 and i + self.window_size < len(df_abnormal):
                if sum(labels[i - 2 * self.window_size:i]) == 0:
                    test_x_lst.append(df_abnormal[i-2*self.window_size:i + self.window_size])
                    label_lst.append(df_label[i-2*self.window_size:i + self.window_size])

        scaler = MinMaxScaler()
        scaler.fit(np.concatenate(x_n_list, axis=0))
        x_n_list = [scaler.transform(i) for i in x_n_list]
        test_x_lst = [scaler.transform(i) for i in test_x_lst]
        self.data_dict['x_n_list'] = np.array(x_n_list)
        if self.shuffle:
            np.random.seed(self.seed)
            indices = np.random.permutation(len(self.data_dict['x_n_list']))
            self.data_dict['x_n_list'] = self.data_dict['x_n_list'][indices]
        self.data_dict['x_ab_list'] = np.array(test_x_lst)
        self.data_dict['label_list'] = np.array(label_lst)

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
