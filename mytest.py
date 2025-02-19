import networkx as nx
import matplotlib.pyplot as plt

def generate_causal_graph(causal_matrix, threshold=0.1, figsize=(8,6), 
                         positive_color='limegreen', negative_color='tomato',
                         node_colors=('skyblue', 'lightgreen'), show_labels=True,
                         title="Causal Graph with Absolute Threshold"):
    """
    生成带权重的一步时延因果图
    
    参数：
    causal_matrix   : list of lists 下三角权重矩阵
    threshold       : float 边创建阈值（绝对值）
    figsize         : tuple 图像尺寸
    positive_color  : str 正向影响边颜色
    negative_color  : str 负向影响边颜色
    node_colors     : tuple (t-1时刻节点颜色, t时刻节点颜色)
    show_labels     : bool 是否显示权重标签
    title           : str 图像标题
    """
    # 参数校验（保持不变）
    assert all(len(row) == len(causal_matrix) for row in causal_matrix), "必须为方阵"
    assert all(causal_matrix[i][j] == 0 for i in range(len(causal_matrix)) 
               for j in range(i+1, len(causal_matrix))), "必须为下三角矩阵"

    N = len(causal_matrix)
    G = nx.DiGraph()

    # 生成节点（保持不变）
    t_minus_1_nodes = [f'X{i}_t-1' for i in range(N)]
    t_nodes = [f'X{i}_t' for i in range(N)]
    G.add_nodes_from(t_minus_1_nodes + t_nodes)

    # 添加边（保持不变）
    edges = []
    for target in range(N):
        for source in range(N):
            weight = causal_matrix[target][source]
            if abs(weight) > threshold:
                edges.append((
                    f'X{source}_t-1',
                    f'X{target}_t',
                    {'weight': round(weight, 2)}
                ))
    G.add_edges_from(edges)

    # 设置布局（保持不变）
    pos = {
        **{f'X{i}_t-1': (0, N-1-i) for i in range(N)},
        **{f'X{i}_t': (2, N-1-i) for i in range(N)}
    }

    # 绘图（保持不变）
    plt.figure(figsize=figsize)
    
    # 绘制节点（保持不变）
    nx.draw_networkx_nodes(
        G, pos,
        node_size=800,
        node_color=[node_colors[0]]*N + [node_colors[1]]*N
    )
    
    # 绘制边
    edge_data = G.edges(data=True)
    edge_colors = []
    for u, v, d in edge_data:
        weight = d['weight']
        edge_colors.append(positive_color if weight > 0 else negative_color)
    
    nx.draw_networkx_edges(
        G, pos,
        edgelist=edge_data,
        edge_color=edge_colors,
        width=2,  # 使用固定的边宽度
        arrowsize=20
    )
    
    # 添加标签（关键修改点2：调整标签位置）
    nx.draw_networkx_labels(G, pos, font_size=10)
    if show_labels:
        edge_labels = {(u, v): d['weight'] for u, v, d in edge_data}
        nx.draw_networkx_edge_labels(
            G, pos,
            edge_labels=edge_labels,
            font_size=8,
            label_pos=0.5  # 从0.75改为中间位置
        )
    
    # 添加标注（保持不变）
    #plt.text(-0.5, N-0.5, "t-1", fontsize=12, ha='center')
    #plt.text(2.5, N-0.5, "t", fontsize=12, ha='center')
    plt.title(title + f"\n(Threshold: |weight| >{threshold})", fontsize=12)
    plt.axis('off')
    
    return G, plt.gcf()
    

# 示例用法（保持不变）
if __name__ == "__main__":
    test_matrix = [
        [0.0,   0,     0],
        [0.3,   0,     0],
        [-0.15, 0.8,   0]
    ]
    
    G, fig = generate_causal_graph(
        causal_matrix=test_matrix,
        threshold=0.1,
        positive_color='dodgerblue',
        negative_color='darkorange',
        title="Dynamic Causal Relationships"
    )
    
    plt.show()
    plt.savefig("window_A")