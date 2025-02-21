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

    # 加载异常数据
    anomaly_data_path = '/home/hz/projects/AERCA/datasets/data_10_26/test_d/data_processed/root_cause_segment.npy'
    anomaly_data = np.load(anomaly_data_path)
    print("异常数据形状:", anomaly_data.shape)  # (1, x, features)

    # # 修改根因定位部分
    # print("\n开始根因定位...")
    # # 将数据转换为列表格式
    # xs = [anomaly_data[0]]  # 转换为列表，每个元素是(x, features)的数组
    
    # # 修改标签创建方式：创建混合的标签（部分正常，部分异常）
    # label = np.zeros((anomaly_data.shape[1], anomaly_data.shape[2]))
    # # 将中间部分标记为异常（假设异常发生在中间段）
    # mid_start = anomaly_data.shape[1] // 4
    # mid_end = mid_start * 3
    # label[mid_start:mid_end, :] = 1
    # labels = [label]  # 转换为列表格式
    
    # try:
    #     # 使用_testing_root_cause进行根因定位
    #     root_causes = rootad_model._testing_root_cause(xs, labels)
        
    #     # 打印根因定位结果
    #     print("\n根因定位结果:")
    #     feature_mapping = rootad_model.load_feature_mapping(os.path.join(output_dir, 'feature_mapping.txt'))
        
    #     # 解析并显示结果
    #     for t, causes in enumerate(root_causes):
    #         if causes:  # 如果在该时间点检测到根因
    #             print(f"\n时间点 {t}:")
    #             for cause in causes:
    #                 feature_name = feature_mapping[f'X{cause}']
    #                 print(f"  根因特征: {feature_name}")
    # except Exception as e:
    #     print(f"根因定位过程出错: {e}")
    #     print("请检查数据和标签的分布情况")


    # 修改根因定位部分
    print("\n开始根因定位...")
    # 将数据转换为列表格式
    xs = [anomaly_data[0]]  # 转换为列表，每个元素是(x, features)的数组
    
    # 修改标签创建方式：创建混合的标签
    label = np.zeros((anomaly_data.shape[1], anomaly_data.shape[2]))
    mid_start = anomaly_data.shape[1] // 4
    mid_end = mid_start * 3
    label[mid_start:mid_end, :] = 1
    labels = [label]
    
    try:
        # 使用_testing_root_cause进行根因定位
        pred_labels = rootad_model._testing_root_cause(xs, labels)
        
        print("\n=== 详细的根因定位结果 ===")
        print(f"预测标签形状: {pred_labels.shape}")
        print(f"预测标签类型: {type(pred_labels)}")
        print("\n检测到的异常:")
        
        # 处理一维数组输出
        num_features = anomaly_data.shape[2]  # 特征数量
        
        # 遍历预测标签
        for i, is_anomaly in enumerate(pred_labels):
            if is_anomaly:
                # 计算对应的时间点和特征
                time_step = i // num_features  # 行（时间点）
                feature_idx = i % num_features  # 列（特征）
                
                print(f"\n时间点(行) {time_step}, 特征(列) {feature_idx}:")
                value = xs[0][time_step][feature_idx]
                print(f"  异常值: {value:.4f}")
        
    except Exception as e:
        print(f"根因定位过程出错: {e}")
        print(f"错误类型: {type(e)}")
        import traceback
        print(f"详细错误信息: {traceback.format_exc()}")

    # 生成因果图
    print("\n生成因果图...")
    output_dir = '/home/hz/projects/AERCA/datasets/data_10_26/test_d/data_processed'

    # 生成真实因果图
    _, fig1 = rootad_model.generate_causal_graph(
        true_causal_struct, 
        os.path.join(output_dir, "true_causal_graph.png"),
        output_dir=output_dir,
        title="True Causal Graph"
    )
    print("真实因果图已保存")

    # 生成预测因果图
    encA_lower = rootad_model.make_lower_triangular(encA[min(9, len(encA)-1)])
    _, fig2 = rootad_model.generate_causal_graph(
        encA_lower, 
        os.path.join(output_dir, "pred_causal_graph.png"),
        output_dir=output_dir,
        title="Predicted Causal Graph"
    )
    print("预测因果图已保存")

if __name__ == '__main__':
    main(sys.argv) 