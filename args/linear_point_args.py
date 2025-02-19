import argparse 

def args_parser():
    parser = argparse.ArgumentParser(description='Linear')
    # Dataset
    parser.add_argument('--T', type=int, default=500, help='Length of the time series (default: 50)')
    parser.add_argument('--training_size', type=int, default=10)
    parser.add_argument('--testing_size', type=int, default=100)
    parser.add_argument('--num_vars', type=int, default=4)
    parser.add_argument('--preprocessing_data', type=int, default=1)
    parser.add_argument('--adlength', type=int, default=1)
    parser.add_argument('--adtype', type=str, default='non_causal')
    parser.add_argument('--mul', type=int, default=3)
    parser.add_argument('--a', type=int, default=None)
    parser.add_argument('--data_dir', type=str, default='./datasets/linear_point')
    parser.add_argument('--causal_quantile', type=float, default=0.50)


    # Meta
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--dataset_name', type=str, default='linear')

    # RootAD
    parser.add_argument('--window_size', type=int, default=1)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--encoder_alpha', type=float, default=0.5)
    parser.add_argument('--decoder_alpha', type=float, default=0.5)
    parser.add_argument('--encoder_gamma', type=float, default=0.5)
    parser.add_argument('--decoder_gamma', type=float, default=0.5)
    parser.add_argument('--encoder_lambda', type=float, default=0.5)
    parser.add_argument('--decoder_lambda', type=float, default=0.5)
    parser.add_argument('--beta', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--epochs', type=int, default=50)   #default=1000
    parser.add_argument('--hidden_layer_size', type=int, default=50)
    parser.add_argument('--num_hidden_layers', type=int, default=1)
    parser.add_argument('--recon_threshold', type=float, default=0.95)
    parser.add_argument('--root_cause_threshold_encoder', type=float, default=0.99)
    parser.add_argument('--root_cause_threshold_decoder', type=float, default=0.99)
    parser.add_argument('--training_rootad', type=int, default=1)
    parser.add_argument('--initial_z_score', type=float, default=3.0)
    parser.add_argument('--risk', type=float, default=1e-2)
    parser.add_argument('--initial_level', type=float, default=0.98)
    parser.add_argument('--num_candidates', type=int, default=100)

    return parser


