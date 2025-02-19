import argparse 

def args_parser():
    parser = argparse.ArgumentParser()

    parser = argparse.ArgumentParser(description='SWaT')
    # Dataset
    parser.add_argument('--preprocessing_data', type=int, default=1)
    parser.add_argument('--data_dir', type=str, default='./datasets/SWaT')
    parser.add_argument('--num_vars', type=int, default=51)
    parser.add_argument('--causal_quantile', type=float, default=0.70)
    parser.add_argument('--shuffle', type=int, default=1)

    # Meta
    parser.add_argument('--seed', type=int, default=2, help='Random seed (default: 42)')
    parser.add_argument('--device', type=str, default='cuda:1')
    parser.add_argument('--dataset_name', type=str, default='SWaT')

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
    parser.add_argument('--lr', type=float, default=0.000001)
    parser.add_argument('--epochs', type=int, default=5000)
    parser.add_argument('--hidden_layer_size', type=int, default=1000)
    parser.add_argument('--num_hidden_layers', type=int, default=8)
    parser.add_argument('--recon_threshold', type=float, default=0.95)
    parser.add_argument('--root_cause_threshold_encoder', type=float, default=0.99)
    parser.add_argument('--root_cause_threshold_decoder', type=float, default=0.99)
    parser.add_argument('--training_rootad', type=int, default=1)
    parser.add_argument('--initial_z_score', type=float, default=3.0)
    parser.add_argument('--risk', type=float, default=1e-5)
    parser.add_argument('--initial_level', type=float, default=0.00)
    parser.add_argument('--num_candidates', type=int, default=100)

    return parser


