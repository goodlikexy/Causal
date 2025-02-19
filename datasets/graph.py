# OCT 2023
# new data generated according to ER-graphs
# for 6 variables only 

import numpy as np
import random
import igraph as ig
import os
from tqdm import tqdm

class GraphTimeSeriesGenerator:
    def __init__(self, options):
        self.options = options
        self.data_dict = {}
        self.seed = options['seed']
        self.n = options['training_size'] + options['testing_size']
        self.t = options['T']
        self.m = options['m']
        self.num_vars = options['num_vars']
        self.data_dir = options['data_dir']
        self.mul = options['mul']
        self.adlength = options['adlength']
        self.adtype = options['adtype']
        self.noise_scale = options['noise_scale']
        self.generate_er_graph()

    def generate_er_graph(self):
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
        G_und = ig.Graph.Erdos_Renyi(n=self.num_vars, m=self.m, directed=True, loops=False)
        self.data_dict['causal_struct'] = np.array(G_und.get_adjacency().data).T
        self.data_dict['signed_causal_struct'] = None

    def generate_example(self):
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)

        x_n_list = []
        x_ab_list = []
        eps_n_list = []
        eps_ab_list = []
        label_list = []

        coefficients = np.random.uniform(low=0.1, high=2.0, size=(self.num_vars, self.num_vars, 5))

        for i in tqdm(range(self.n)):
            eps = self.noise_scale * np.random.randn(self.t, self.num_vars)
            x = np.zeros((self.t, self.num_vars))
            x[:5] = np.random.randn(5*self.num_vars).reshape(5, self.num_vars)
            x_ab = np.zeros((self.t, self.num_vars))
            x_ab[:5] = x[:5]

            for t in range(5, self.t):
                for var in range(self.num_vars):
                    for j in range(self.num_vars):
                        if self.data_dict['causal_struct'][var, j] == 1:
                            for lag in range(1, 6):
                                x[t, var] += coefficients[var, j, lag-1] * np.cos(x[t-lag, j] + 1)
                    x[t, var] += eps[t, var]

            x_n_list.append(x)
            eps_n_list.append(eps)

            t_p = np.random.randint(int(0.2 * self.t), int(0.8 * self.t), size=1)
            if self.adlength > 1:
                temp_t_p = []
                for j in range(self.adlength):
                    temp_t_p.append(t_p+j)
                t_p = np.array(temp_t_p)
            feature_p = np.random.permutation(np.arange(self.num_vars))[:np.random.randint(1, min(10, self.num_vars)+1)]
            ab = np.zeros(self.num_vars)
            ab[feature_p] += self.mul
            temp_label = np.zeros((self.t, self.num_vars))
            temp_label[t_p, feature_p] += 1

            for t in range(5, self.t):
                if t in t_p:
                    if self.adtype == 'non_causal':
                        eps[t] += ab
                        for var in range(self.num_vars):
                            for j in range(self.num_vars):
                                if self.data_dict['causal_struct'][var, j] == 1:
                                    for lag in range(1, 6):
                                        x_ab[t, var] += coefficients[var, j, lag-1] * np.cos(x_ab[t-lag, j] + 1)
                            x_ab[t, var] += eps[t, var]

                    elif self.adtype == 'causal':
                        raise NotImplementedError("Causal anomaly not implemented for this dataset.")
                    else:
                        raise NotImplementedError("Invalid adtype. Expected 'non_causal' or 'causal'.")
                else:
                    for var in range(self.num_vars):
                        for j in range(self.num_vars):
                            if self.data_dict['causal_struct'][var, j] == 1:
                                for lag in range(1, 6):
                                    x_ab[t, var] += coefficients[var, j, lag-1] * np.cos(x_ab[t-lag, j] + 1)
                        x_ab[t, var] += eps[t, var]
            x_ab_list.append(x_ab)
            eps_ab_list.append(eps)
            label_list.append(temp_label)

            self.data_dict['x_n_list'] = np.array(x_n_list)
            self.data_dict['x_ab_list'] = np.array(x_ab_list)
            self.data_dict['eps_n_list'] = np.array(eps_n_list)
            self.data_dict['eps_ab_list'] = np.array(eps_ab_list)
            self.data_dict['label_list'] = np.array(label_list)

    def save_data(self):
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

        np.save(os.path.join(self.data_dir, 'x_n_list'), self.data_dict['x_n_list'])
        np.save(os.path.join(self.data_dir, 'x_ab_list'), self.data_dict['x_ab_list'])
        np.save(os.path.join(self.data_dir, 'eps_n_list'), self.data_dict['eps_n_list'])
        np.save(os.path.join(self.data_dir, 'eps_ab_list'), self.data_dict['eps_ab_list'])
        np.save(os.path.join(self.data_dir, 'causal_struct'), self.data_dict['causal_struct'])
        np.save(os.path.join(self.data_dir, 'label_list'), self.data_dict['label_list'])
        print(self.data_dict['causal_struct'])

    def load_data(self):
        self.data_dict['x_n_list'] = np.load(os.path.join(self.data_dir, 'x_n_list.npy'))
        self.data_dict['x_ab_list'] = np.load(os.path.join(self.data_dir, 'x_ab_list.npy'))
        self.data_dict['eps_n_list'] = np.load(os.path.join(self.data_dir, 'eps_n_list.npy'))
        self.data_dict['eps_ab_list'] = np.load(os.path.join(self.data_dir, 'eps_ab_list.npy'))
        self.data_dict['causal_struct'] = np.load(os.path.join(self.data_dir, 'causal_struct.npy'))
        self.data_dict['label_list'] = np.load(os.path.join(self.data_dir, 'label_list.npy'))
        self.data_dict['signed_causal_struct'] = None
        print(self.data_dict['causal_struct'])



