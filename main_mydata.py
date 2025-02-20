from datasets import linear
import logging
import os
from models import rootad
from args import linear_point_args
import sys
from utils import utils
import numpy as np
from matplotlib import pyplot as plt
import torch

def main(argv):
    # 设置日志
    logging_dir = '/home/hz/projects/AERCA/logs'
    if not os.path.exists(logging_dir):
        os.makedirs(logging_dir)
    log_file_path = os.path.join(logging_dir, 'normal_data_test.log')
    logging.basicConfig(filename=log_file_path,
                    filemode='w',
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

    # 解析参数
    parser = linear_point_args.args_parser()
    args, unknown = parser.parse_known_args()
    options = vars(args)

    # 设置随机种子
    utils.set_seed(options['seed'])
    logging.info('Seed: {}'.format(options['seed']))

    # 加载正常数据
    data_path = '/home/hz/projects/AERCA/datasets/data_10_26/test_d/data_processed/normal_segments.npy'
    normal_data = np.load(data_path)
    print("数据形状:", normal_data.shape)  # (n, 50, features)

    # 准备训练和测试数据
    train_size = int(0.8 * len(normal_data))  # 80% 用于训练
    x_train = normal_data[:train_size]
    x_test = normal_data[train_size:]

    # 创建一个假的因果结构矩阵（如果没有真实的因果结构）
    num_features = normal_data.shape[2]
    dummy_causal_struct = np.zeros((num_features, num_features))

    # 创建 RootAD 模型
    rootad_model = rootad.RootAD(
        num_vars=normal_data.shape[2],  # 特征数量
        hidden_layer_size=options['hidden_layer_size'],
        num_hidden_layers=options['num_hidden_layers'],
        device=options['device'],
        window_size=options['window_size'],
        stride=options['stride'],
        encoder_gamma=options['encoder_gamma'],
        decoder_gamma=options['decoder_gamma'],
        encoder_lambda=options['encoder_lambda'],
        decoder_lambda=options['decoder_lambda'],
        beta=options['beta'],
        lr=options['lr'],
        epochs=options['epochs'],
        recon_threshold=options['recon_threshold'],
        data_name=options['dataset_name'],
        causal_quantile=options['causal_quantile'],
        root_cause_threshold_encoder=options['root_cause_threshold_encoder'],
        root_cause_threshold_decoder=options['root_cause_threshold_decoder'],
        risk=options['risk'],
        initial_level=options['initial_level'],
        num_candidates=options['num_candidates']
    )

    # 训练模型
    if options['training_rootad']:
        rootad_model._training(x_train)
        print('训练完成')

    # 创建全1的下三角矩阵作为真实因果结构
    num_features = normal_data.shape[2]
    true_causal_struct = np.tril(np.ones((num_features, num_features)), k=-1)
    print("\n构造的真实因果结构矩阵:")
    print(true_causal_struct)

    # 使用 _testing_causal_discover 方法
    encA, decA = rootad_model._testing_causal_discover(x_test, true_causal_struct)

    print("\nencA.shape:", encA.shape)
    print("decA.shape:", decA.shape)

    # 打印一些结果
    print("\n编码器因果矩阵示例:")
    print("A[1]:\n", encA[1])
    if len(encA) > 50:
        print("A[50]:\n", encA[50])
    if len(encA) > 99:
        print("A[99]:\n", encA[99])
    
    print("\n解码器因果矩阵示例:")
    print("A[1]:\n", decA[1])
    if len(decA) > 50:
        print("A[50]:\n", decA[50])
    if len(decA) > 99:
        print("A[99]:\n", decA[99])

    # 生成真实因果图和预测因果图
    print("\n生成因果图...")
    _, fig1 = rootad_model.generate_causal_graph(true_causal_struct, "true_causal_graph.png")
    print("真实因果图已保存")

    encA_lower = rootad_model.make_lower_triangular(encA[min(9, len(encA)-1)])
    _, fig2 = rootad_model.generate_causal_graph(encA_lower, "pred_causal_graph.png")
    print("预测因果图已保存")

if __name__ == '__main__':
    main(sys.argv) 