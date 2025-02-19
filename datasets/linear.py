# Some synthetic datasets with linear dynamics
import numpy as np
import os 


class LinearDynamics:
    def __init__(self, options):
        """
            This function generates synthetic datasets with linear dynamics.

            Parameters:
            n (int): Number of replicates.
            t (int): Length of time series.
            mul (int, optional): Multiplier for the anomaly. Default is 10.
            a (array, optional): Coefficients for the linear dynamics. If not provided, random coefficients are generated.
            seed (int, optional): Seed for the random number generator. Default is 0.
            adlength (int, optional): Length of the anomaly. Default is 1.
            adtype (str, optional): Type of anomaly. Can be 'non_causal' or 'causal'. Default is 'non_causal'.
        """
        self.options = options
        self.data_dict = {}
        self.seed = options['seed']
        self.n = options['training_size'] + options['testing_size']
        self.t = options['T']
        self.mul = options['mul']
        self.a = options['a'] if options['a'] is not None else self._generate_random_coefficients()
        self.adlength = options['adlength']
        self.adtype = options['adtype']
        self.data_dir = options['data_dir']

    def _generate_random_coefficients(self):
        a = np.zeros((8,))
        for k in range(8):
            u_1 = np.random.uniform(0, 1, 1)
            a[k] = np.random.uniform(-0.8, -0.2, 1) if u_1 <= 0.5 else np.random.uniform(0.2, 0.8, 1)
        return a
    def generate_example(self):
        if self.seed is not None:
            np.random.seed(self.seed)

        # Initialize lists to store results
        x_n_list = []
        x_ab_list = []
        eps_n_list = []
        eps_ab_list = []
        label_list = []

        # Generate data for each sample
        for i in range(self.n):
            # Generate random noise for each variable
            eps_x = 0.4 * np.random.normal(0, 1, (self.t,))
            eps_y = 0.4 * np.random.normal(0, 1, (self.t,))
            eps_w = 0.4 * np.random.normal(0, 1, (self.t,))
            eps_z = 0.4 * np.random.normal(0, 1, (self.t,))

            # Initialize variables
            x = np.zeros((self.t, 1))
            y = np.zeros((self.t, 1))
            w = np.zeros((self.t, 1))
            z = np.zeros((self.t, 1))

            # Generate time series data
            for j in range(1, self.t):
                x[j, 0] = self.a[0] * x[j - 1, 0] + eps_x[j]
                w[j, 0] = self.a[1] * w[j - 1, 0] + self.a[2] * x[j - 1, 0] + eps_w[j]
                y[j, 0] = self.a[3] * y[j - 1, 0] + self.a[4] * w[j - 1, 0] + eps_y[j]
                z[j, 0] = self.a[5] * z[j - 1, 0] + self.a[6] * w[j - 1, 0] + self.a[7] * y[j - 1, 0] + eps_z[j]

            # Store results
            x_n_list.append(np.concatenate((x, w, y, z), axis=1))
            eps_n_list.append(np.concatenate((eps_x, eps_w, eps_y, eps_z), axis=0).reshape(4,-1).T)
            t_p = np.random.randint(int(0.2 * self.t), int(0.8 * self.t), size=1)
            if self.adlength > 1:
                temp_t_p = []
                for j in range(self.adlength):
                    temp_t_p.append(t_p+j)
                t_p = np.array(temp_t_p)

            feature_p = np.random.permutation(np.arange(4))[:np.random.randint(1,5)]
            ab = np.zeros(4)
            ab[feature_p] += self.mul
            temp_label = np.zeros((self.t, 4))
            if len(t_p) > 1:
                for temp_t_p, temp_feature_p in zip(t_p, feature_p):
                    temp_label[temp_t_p, temp_feature_p] += 1
            else:
                temp_label[t_p, feature_p] += 1

            # Initialize variables for anomaly data
            x = np.zeros((self.t, 1))
            y = np.zeros((self.t, 1))
            w = np.zeros((self.t, 1))
            z = np.zeros((self.t, 1))

            # Generate anomaly time series data
            for j in range(1, self.t):
                if j in t_p:
                    if self.adtype == 'non_causal':
                        eps_x[j] += ab[0]
                        eps_w[j] += ab[1]
                        eps_y[j] += ab[2]
                        eps_z[j] += ab[3]
                        x[j, 0] = self.a[0] * x[j - 1, 0] + eps_x[j]
                        w[j, 0] = self.a[1] * w[j - 1, 0] + self.a[2] * x[j - 1, 0] + eps_w[j]
                        y[j, 0] = self.a[3] * y[j - 1, 0] + self.a[4] * w[j - 1, 0] + eps_y[j]
                        z[j, 0] = self.a[5] * z[j - 1, 0] + self.a[6] * w[j - 1, 0] + self.a[7] * y[j - 1, 0] + eps_z[j]
                    elif self.adtype == 'causal':
                        b = self.a.copy()*3

                        z[j, 0] = b[0] * z[j - 1, 0] + eps_z[j]
                        y[j, 0] = b[1] * y[j - 1, 0] + b[2] * x[j - 1, 0] + eps_y[j]
                        w[j, 0] = b[3] * w[j - 1, 0] + b[4] * w[j - 1, 0] + eps_w[j]
                        x[j, 0] = b[5] * x[j - 1, 0] + b[6] * w[j - 1, 0] + b[7] * y[j - 1, 0] + eps_x[j]

                    else:
                        raise NotImplementedError("Invalid adtype. Expected 'non_causal' or 'causal'.")
                else:
                    x[j, 0] = self.a[0] * x[j - 1, 0] + eps_x[j]
                    w[j, 0] = self.a[1] * w[j - 1, 0] + self.a[2] * x[j - 1, 0] + eps_w[j]
                    y[j, 0] = self.a[3] * y[j - 1, 0] + self.a[4] * w[j - 1, 0] + eps_y[j]
                    z[j, 0] = self.a[5] * z[j - 1, 0] + self.a[6] * w[j - 1, 0] + self.a[7] * y[j - 1, 0] + eps_z[j]

            # Store anomaly results
            x_ab_list.append(np.concatenate((x, w, y, z), axis=1))
            eps_ab_list.append(np.concatenate((eps_x, eps_w, eps_y, eps_z), axis=0).reshape(4, -1).T)
            label_list.append(temp_label)

        # Define causal sturcture
        causal_struct_value = np.array([[self.a[0], 0   , 0   , 0   ],
                                         [self.a[2], self.a[1], 0   , 0   ],
                                         [0   , self.a[4], self.a[3], 0   ],
                                         [0   , self.a[6], self.a[7], self.a[5]]])
        a_signed = np.sign(self.a)
        signed_causal_struct = np.array([[a_signed[0], 0, 0, 0],
                                         [a_signed[2], a_signed[1], 0, 0],
                                         [0, a_signed[4], a_signed[3], 0],
                                         [0, a_signed[6], a_signed[7], a_signed[5]]])
        causal_struct = np.array([[1, 0, 0, 0],
                                  [1, 1, 0, 0],
                                  [0, 1, 1, 0],
                                  [0, 1, 1, 1]])

        # Put results into a dictionary
        self.data_dict['x_n_list'] = np.array(x_n_list)
        self.data_dict['x_ab_list'] = np.array(x_ab_list)
        self.data_dict['eps_n_list'] = np.array(eps_n_list)
        self.data_dict['eps_ab_list'] = np.array(eps_ab_list)
        self.data_dict['causal_struct'] = causal_struct
        self.data_dict['causal_struct_value'] = causal_struct_value
        self.data_dict['signed_causal_struct'] = signed_causal_struct
        self.data_dict['label_list'] = np.array(label_list)
        self.data_dict['a'] = self.a

    def save_data(self):
        # Create the directory if it does not exist
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

        # Save the data
        np.save(os.path.join(self.data_dir, 'x_n_list.npy'), self.data_dict['x_n_list'])
        np.save(os.path.join(self.data_dir, 'x_ab_list.npy'), self.data_dict['x_ab_list'])
        np.save(os.path.join(self.data_dir, 'eps_n_list.npy'), self.data_dict['eps_n_list'])
        np.save(os.path.join(self.data_dir, 'eps_ab_list.npy'), self.data_dict['eps_ab_list'])
        np.save(os.path.join(self.data_dir, 'causal_struct.npy'), self.data_dict['causal_struct'])
        np.save(os.path.join(self.data_dir, 'causal_struct_value.npy'), self.data_dict['causal_struct_value'])
        np.save(os.path.join(self.data_dir, 'signed_causal_struct.npy'), self.data_dict['signed_causal_struct'])
        np.save(os.path.join(self.data_dir, 'label_list.npy'), self.data_dict['label_list'])
        np.save(os.path.join(self.data_dir, 'a.npy'), self.data_dict['a'])

    def load_data(self):
        self.data_dict['x_n_list'] = np.load(os.path.join(self.data_dir, 'x_n_list.npy'))
        self.data_dict['x_ab_list'] = np.load(os.path.join(self.data_dir, 'x_ab_list.npy'))
        self.data_dict['eps_n_list'] = np.load(os.path.join(self.data_dir, 'eps_n_list.npy'))
        self.data_dict['eps_ab_list'] = np.load(os.path.join(self.data_dir, 'eps_ab_list.npy'))
        self.data_dict['causal_struct'] = np.load(os.path.join(self.data_dir, 'causal_struct.npy'))
        self.data_dict['causal_struct_value'] = np.load(os.path.join(self.data_dir, 'causal_struct_value.npy'))
        self.data_dict['signed_causal_struct'] = np.load(os.path.join(self.data_dir, 'signed_causal_struct.npy'))
        self.data_dict['label_list'] = np.load(os.path.join(self.data_dir, 'label_list.npy'))
        self.data_dict['a'] = np.load(os.path.join(self.data_dir, 'a.npy'))

