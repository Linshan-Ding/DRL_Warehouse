# DRL_Warehouse

Paper Title: **The Impact of Dynamic Workforce Allocation in Human-Robot Collaborative Warehouses: A Human-Centric Perspective**

本项目用于研究人机协同仓库中的动态资源配置问题。仓库环境包含订单随机到达、机器人搬运、拣货员拣选、订单延期惩罚、短租/长租成本等要素，并通过 PPO（Proximal Policy Optimization）训练智能体，比较不同租赁策略下的运行成本与订单完成表现。

## 功能概览

- 构建多区域、多巷道、多货架层的仓库仿真环境。
- 生成基于泊松到达过程的订单算例，并保存为 `.pkl` 文件。
- 使用 PPO 训练三种资源配置策略：
  - `PPO_I.py`：短租模式，按天动态调整机器人和各区域拣货员数量。
  - `PPO_II.py`：长租模式，按月或整期配置资源。
  - `PPO_III.py`：长短租结合模式，支持长期基础资源与短期补充资源协同。
- 记录总成本、延期成本、机器人成本、拣货员成本、订单完成率、平均拣选时间等指标。
- 通过 Visdom 实时查看训练过程中的成本曲线。

## 项目结构

```text
DRL_Warehouse/
├── README.md
├── agent/
│   ├── PPO_I.py                 # 短租 PPO 训练脚本
│   ├── PPO_II.py                # 长租 PPO 训练脚本
│   └── PPO_III.py               # 长短租结合 PPO 训练脚本
├── data/
│   ├── generat_order_lamdas.py  # 订单算例生成脚本
│   ├── warehouse.py             # 用于辅助生成算例的仓库环境
│   └── instances/               # 订单算例文件，例如 orders_2.pkl
├── environment/
│   ├── class_public.py          # 环境、成本、订单和 PPO 参数配置
│   ├── class_warehouse.py       # 订单、商品、储货位、拣货位、机器人、拣货员等实体类
│   ├── warehouse_test2.py       # 当前 PPO 脚本主要使用的仿真环境
│   ├── warehouse_I.py           # 短租实验环境
│   ├── warehouse_II.py          # 长租实验环境
│   ├── warehouse_III.py         # 长短租结合实验环境
│   └── warehouse_III_pygame*.py # Pygame 可视化实验版本
├── test/                        # 实验/测试脚本
└── result/                      # 结果文件目录
```

## 环境准备

建议在项目根目录创建独立虚拟环境：

```powershell
cd "D:\Python project\DRL_Warehouse"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch numpy gymnasium visdom pygame
```

训练脚本会连接 Visdom。如果需要实时曲线展示，请先在另一个终端启动：

```powershell
python -m visdom.server
```

默认访问地址为 `http://localhost:8097`。

## 数据准备

订单算例存放在 `data/instances/`，文件名格式为：

```text
orders_<单个订单商品数>.pkl
```

生成示例数据：

```powershell
python data/generat_order_lamdas.py
```

当前生成脚本中的 `order_n_items_list` 默认包含 `[2, 6]`。如果训练脚本使用 `current_scenario = 10`，请先生成 `orders_10.pkl`，或将对应 PPO 脚本中的 `current_scenario` 修改为已有算例编号。

## 运行训练

在项目根目录运行对应策略脚本：

```powershell
python agent\PPO_I.py
python agent\PPO_II.py
python agent\PPO_III.py
```

脚本末尾默认调用：

```python
train_ppo_agent(ppo_agent, warehouse, num_episodes=3000)
```

如需快速验证流程，可先将 `num_episodes` 调小。每个动作向量通常表示：

```text
[机器人数量, 区域1拣货员数量, 区域2拣货员数量, 区域3拣货员数量]
```

## 主要配置

核心参数集中在 `environment/class_public.py`：

- `warehouse`：货架容量、货架层数、区域数量、巷道数量、入口位置等。
- `robot`：短租/长租机器人运行成本、移动速度。
- `picker`：短租/长租拣货员成本、移动速度、辞退成本。
- `order`：订单延期成本、打包时间、到达率、交期、订单商品数范围。
- `item`：商品拣选时间。
- `ppo`：折扣因子、学习率、clip range、熵系数等 PPO 超参数。

训练轮数、batch size、算例编号、模型结构等参数主要位于 `agent/PPO_*.py`。

## 输出文件

训练过程中会生成以下文件：

- `instance_data_PPO_*_all.csv`：全部算例的训练指标。
- `instance_data_PPO_*_<scenario>.csv`：指定算例的训练指标。
- `daily_configurations_PPO_*.csv`：短租或长短租策略的每日资源配置。
- `monthly_configurations_PPO_II.csv`：长租策略的月度资源配置。
- `policy_network_PPO_*.pth`：当前最优策略网络权重。
- `../result/result_file/training_data_PPO_*.csv`：脚本中写入的简要训练日志。

注意：部分脚本仍包含硬编码路径，例如 `D:/Python project/DRL_Warehouse/data/instances` 和 `../result/result_file/`。如果移动项目位置，或输出目录不存在，需要同步调整这些路径或提前创建对应目录。

## 常见问题

**找不到 `orders_10.pkl`**

默认 PPO 脚本中可能强制使用 `current_scenario = 10`。请生成对应算例，或把该值改为 `2`、`6` 等已存在的算例编号。

**Visdom 连接失败**

先运行 `python -m visdom.server`。如果只想离线训练，也可以暂时忽略可视化告警，但训练曲线不会显示在浏览器中。

**导入路径异常**

请从项目根目录运行脚本，保证 `agent/`、`data/`、`environment/` 能被 Python 正确识别。部分历史实验脚本可能保留旧导入路径，当前 PPO 训练入口主要以 `environment/warehouse_test2.py` 为准。

## 推荐实验流程

1. 在 `environment/class_public.py` 中确认仓库、订单和成本参数。
2. 运行 `data/generat_order_lamdas.py` 生成所需订单算例。
3. 根据研究场景选择 `PPO_I.py`、`PPO_II.py` 或 `PPO_III.py`。
4. 启动 Visdom 并运行训练脚本。
5. 对比 CSV 输出中的总成本、资源成本、延期成本、订单完成率和平均拣选时间。
