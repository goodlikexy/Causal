from datasets import linear, multiple_lotka_volterra
import logging
import os
from models import rootad
from args import linear_point_args, lv_point_args
import sys
from utils import utils
import numpy as np
from matplotlib import pyplot as plt 

def main(argv):
    logging_dir = '/home/hz/projects/AERCA/logs'
    # Create the directory if it does not exist
    if not os.path.exists(logging_dir):
        os.makedirs(logging_dir)

    # Full path for the log file
    log_file_path = os.path.join(logging_dir, 'linear.log')
    # Set up logging
    logging.basicConfig(filename=log_file_path,
                    filemode='w', # Append mode (use 'w' for write mode)
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

    # Parse command line arguments
    parser = linear_point_args.args_parser()
    args, unknown = parser.parse_known_args()
    options = vars(args)

    # Set the random seed
    utils.set_seed(options['seed'])
    logging.info('Seed: {}'.format(options['seed']))

    # If preprocessing data, generate data and save it
    # Otherwise, load the data
    data_class = linear.LinearDynamics(options)
    if options['preprocessing_data'] == 1:
        logging.info('Preprocessing data')
        # Generate data
        data_class.generate_example()
        data_class.save_data()
        # Save data
    else:
        logging.info('Loading data')
        # Load data
        data_class.load_data()

    rootad_model = rootad.RootAD(num_vars=options['num_vars'], hidden_layer_size=options['hidden_layer_size'],
                          num_hidden_layers=options['num_hidden_layers'], device=options['device'],
                          window_size=options['window_size'], stride=options['stride'],
                          encoder_gamma=options['encoder_gamma'], decoder_gamma=options['decoder_gamma'],
                          encoder_lambda=options['encoder_lambda'], decoder_lambda=options['decoder_lambda'],
                          beta=options['beta'], lr=options['lr'], epochs=options['epochs'],
                          recon_threshold=options['recon_threshold'], data_name=options['dataset_name'],
                          causal_quantile=options['causal_quantile'], root_cause_threshold_encoder=options['root_cause_threshold_encoder'],
                          root_cause_threshold_decoder=options['root_cause_threshold_decoder'],
                          risk=options['risk'], initial_level=options['initial_level'], num_candidates=options['num_candidates'])
    if options['training_rootad']:
        # Train RootAD
        rootad_model._training(data_class.data_dict['x_n_list'][:options['training_size']])
        print('done training')
    # Test RootAD
    pred_label = rootad_model._testing_root_cause(data_class.data_dict['x_ab_list'][options['training_size']:],
                          data_class.data_dict['label_list'][options['training_size']:])
    encA,decA = rootad_model._testing_causal_discover(data_class.data_dict['x_n_list'][options['training_size']:],
                         data_class.data_dict['causal_struct'])
    print("pred_label:" , pred_label)
    print("encA.shape:" , encA.shape)
    print("encA.shape:" , decA.shape)
    print("encoder A[1]:")
    print(encA[1])
    print("encoder A[50]:")
    print(encA[50])
    print("encoder A[99]:")
    print(encA[99])
    print("````````````````````````````````````")
    print("decoder A[1]:")
    print(decA[1])
    print("decoder A[50]:")
    print(decA[50])
    print("decoder A[99]:")
    print(decA[99])

    print('done')
    print("True causal graph:")
    _, fig1 = rootad_model.generate_causal_graph(data_class.data_dict['causal_struct'], "true_causal_graph.png")
    # plt.savefig("true_causal_graph.png")
    # plt.close()
    
    # 将encA[9]转换为下三角矩阵 
    encA_lower = rootad_model.make_lower_triangular(encA[9])
    print("pred causal graph:")
    _, fig2 = rootad_model.generate_causal_graph(encA_lower, "pred_causal_graph.png")
    # plt.savefig("pred_causal_graph.png")
    # plt.close()


if __name__ == '__main__':
    main(sys.argv)

