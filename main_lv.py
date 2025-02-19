from datasets import linear, multiple_lotka_volterra
import logging
import os
from models import rootad
from args import linear_point_args, lv_point_args
import sys
from utils import utils
import numpy as np 

def main(argv):
    logging_dir = '/home/hz/projects/AERCA/logs'
    # Create the directory if it does not exist
    if not os.path.exists(logging_dir):
        os.makedirs(logging_dir)

    # Full path for the log file
    log_file_path = os.path.join(logging_dir, 'lv.log')
    # Set up logging
    logging.basicConfig(filename=log_file_path,
                    filemode='w', # Append mode (use 'w' for write mode)
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

    # Parse command line arguments
    parser = lv_point_args.args_parser()
    args, unknown = parser.parse_known_args()
    options = vars(args)

    # Set the random seed
    utils.set_seed(options['seed'])
    logging.info('Seed: {}'.format(options['seed']))

    # If preprocessing data, generate data and save it
    # Otherwise, load the data
    data_class = multiple_lotka_volterra.MultiLotkaVolterra(options)
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
    rootad_model._testing_root_cause(data_class.data_dict['x_ab_list'][options['training_size']:],
                          data_class.data_dict['label_list'][options['training_size']:])
    rootad_model._testing_causal_discover(data_class.data_dict['x_n_list'][options['training_size']:],
                         data_class.data_dict['causal_struct'])


    print('done')


if __name__ == '__main__':
    main(sys.argv)

