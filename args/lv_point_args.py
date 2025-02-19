import argparse 

def args_parser():
    parser = argparse.ArgumentParser(description='LV')
    # Dataset
    parser.add_argument('--p', type=int, default=20, help='Number of variables (default: 20)')
    parser.add_argument('--T', type=int, default=20000, help='Length of the time series (default: 500)')
    parser.add_argument('--d', type=int, default=2,
                        help='Number of species hunted and hunted by, in the Lotka-Volterra system (default: 2)')
    parser.add_argument('--dt', type=float, default=0.01, help='Sampling time (default: 0.01)')
    parser.add_argument('--downsample_factor', type=int, default=10, help='Down-sampling factor (default: 10)')
    parser.add_argument('--alpha_lv', type=float, default=1.1,
                        help='Parameter alpha in Lotka-Volterra equations (default: 1.1)')
    parser.add_argument('--beta_lv', type=float, default=0.2,
                        help='Parameter beta in Lotka-Volterra  (default: 0.4)')
    parser.add_argument('--gamma_lv', type=float, default=1.1,
                        help='Parameter gamma in Lotka-Volterra equations (default: 0.4)')
    parser.add_argument('--delta_lv', type=float, default=0.2,
                        help='Parameter delta in Lotka-Volterra equations (default: 0.1)')
    parser.add_argument('--sigma_lv', type=float, default=0.1,
                        help='Noise scale parameter in Lotka-Volterra simulations (default: 0.1)')
    parser.add_argument('--training_size', type=int, default=10)
    parser.add_argument('--testing_size', type=int, default=100)
    parser.add_argument('--preprocessing_data', type=int, default=1)
    parser.add_argument('--adlength', type=int, default=1)
    parser.add_argument('--adtype', type=str, default='non_causal')
    parser.add_argument('--data_dir', type=str, default='./datasets/lv_point')
    parser.add_argument('--num_vars', type=int, default=40)
    parser.add_argument('--mul', type=int, default=20)
    parser.add_argument('--causal_quantile', type=float, default=0.90)

    # Meta
    parser.add_argument('--seed', type=int, default=0, help='Random seed (default: 42)')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--dataset_name', type=str, default='LV')

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
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--hidden_layer_size', type=int, default=50)
    parser.add_argument('--num_hidden_layers', type=int, default=2)
    parser.add_argument('--recon_threshold', type=float, default=0.95)
    parser.add_argument('--root_cause_threshold_encoder', type=float, default=0.99)
    parser.add_argument('--root_cause_threshold_decoder', type=float, default=0.99)
    parser.add_argument('--training_rootad', type=int, default=1)
    parser.add_argument('--initial_z_score', type=float, default=3.0)
    parser.add_argument('--risk', type=float, default=1e-2)
    parser.add_argument('--initial_level', type=float, default=0.98)
    parser.add_argument('--num_candidates', type=int, default=100)

    return parser


