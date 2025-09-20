"""
策略网络结构示意图
"""
from graphviz import Digraph
import torch


def visualize_policy_network():
    # 创建有向图
    dot = Digraph(comment='Policy Network Structure',
                  graph_attr={'rankdir': 'LR', 'splines': 'ortho'},
                  node_attr={'shape': 'record', 'fontsize': '10'})

    # 输入层
    dot.node('matrix_in', 'Matrix Input\n[Channel×H×W]\nInput Channels: 3\nHeight: N_w\nWidth: N_l')
    dot.node('scalar_in', 'Scalar Input\n[Scalar Dim]\nDimension: N_a+1')

    # CNN层
    with dot.subgraph(name='cluster_cnn') as c:
        c.attr(label='CNN Feature Extractor', color='blue', style='rounded')
        c.node('conv1', 'Conv2D\nKernel: 3×3\nStride: 1\nPadding: 1\nChannels: 2')
        c.node('relu1', 'ReLU')
        c.node('conv2', 'Conv2D\nKernel: 3×3\nStride: 1\nPadding: 1\nChannels: 1')
        c.node('relu2', 'ReLU')
        c.node('pool', 'MaxPool2D\nKernel: 2×2\nStride: 2')
        c.node('reshape', 'Flatten')
        c.node('fc0', 'Linear\nInput: Conv Output\nOutput: 3')

        # CNN内部连接
        c.edges([('conv1', 'relu1'), ('relu1', 'conv2'),
                 ('conv2', 'relu2'), ('relu2', 'pool'),
                 ('pool', 'reshape'), ('reshape', 'fc0')])

    # 特征融合
    dot.node('concat', 'Feature Fusion\nConcat(Visual, Scalar)')
    dot.node('norm', 'Layer Normalization')

    # FC模块
    with dot.subgraph(name='cluster_fc') as f:
        f.attr(label=f'FC Blocks\n{Layers} Layers', color='green', style='rounded')
        for i in range(Layers):
            in_dim = 'Input Size' if i == 0 else 'Hidden Dim'
            out_dim = 'Hidden Dim' if i < Layers - 1 else 'Attention Dim'
            f.node(f'fc_{i}', f'Linear {i + 1}\nInput: {in_dim}\nOutput: {out_dim}')
            if i < Layers - 1:
                f.node(f'relu_{i}', 'ReLU')

    # 注意力机制
    with dot.subgraph(name='cluster_attn') as a:
        a.attr(label='Attention Module', color='purple', style='rounded')
        a.node('attn_input', 'Reshape\n[Batch, 1, Features]')
        a.node('multihead', f'MultiheadAttention\nHeads: 4\nEmbed Dim: Hidden Dim')
        a.node('attn_output', 'Reshape\n[Batch, Features]')
        a.node('attn_norm', 'Layer Normalization')

        # 内部连接
        a.edges([('attn_input', 'multihead'), ('multihead', 'attn_output'),
                 ('attn_output', 'attn_norm')])

    # 特征融合与输出
    dot.node('fuse', 'Feature Fusion\nConcat(FC, Attention)')
    dot.node('adapter', 'Adapter\nLinear → Tanh\nInput: 256\nOutput: 128')
    dot.node('mean', 'Linear\nAction Mean\nOutput: N_a+1')
    dot.node('std', 'Action Std\nLearnable Parameter')

    # 完整连接流程
    dot.edges([
        ('matrix_in', 'conv1'),
        ('scalar_in', 'concat'),
        ('fc0', 'concat'),
        ('concat', 'norm'),
        ('norm', 'fc_0')
    ])

    # FC层内部连接
    for i in range(Layers - 1):
        if i < Layers - 2:
            dot.edge(f'fc_{i}', f'relu_{i}')
            dot.edge(f'relu_{i}', f'fc_{i + 1}')
        else:
            dot.edge(f'fc_{i}', f'fc_{i + 1}')

    # 注意力机制连接
    dot.edge(f'fc_{Layers - 1}', 'attn_input')
    dot.edge(f'fc_{Layers - 1}', 'fuse')
    dot.edge('attn_norm', 'fuse')

    # 输出层连接
    dot.edges([
        ('fuse', 'adapter'),
        ('adapter', 'mean'),
        ('mean', 'mean_out'),
        ('std', 'std_out')
    ])

    dot.node('mean_out', 'Action Mean Output')
    dot.node('std_out', 'Action Std Output')

    # 返回图片
    dot.format = 'png'
    dot.render('policy_network', view=True)


# 参数配置
Layers = 10  # 全连接层数量
visualize_policy_network()