# DRL_Warehouse

Paper Title: **The Impact of Dynamic Workforce Allocation in Human-Robot Collaborative Warehouses: A Human-Centric Perspective**

本项目用于研究人机协同仓库中的动态资源配置问题。当前代码已收敛为统一配置、统一环境、统一 PPO 训练入口，并支持按“订单商品数 × 月份”组织训练算例。

## Project Structure

```text
DRL_Warehouse/
├── configs/
│   └── default.json             # 仓库、成本、订单、PPO、实验和路径配置
├── agent/
│   ├── ppo/                     # PPO 统一训练入口与网络
│   └── baselines/               # A2C / DDPG / TD3 / SAC 对比算法
├── data/
│   ├── generate_orders.py       # 订单算例生成入口
│   └── instances/               # items_<n>/orders_m<month>.pkl
├── environment/
│   ├── class_public.py          # 配置加载与校验
│   ├── class_warehouse.py       # 订单、商品、机器人、拣货员等实体
│   └── warehouse_env.py         # 主仿真环境
├── test/                        # 轻量测试
└── result/                      # 实验结果
```

## Setup

```powershell
cd DRL_Warehouse
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch numpy gymnasium visdom pygame
```

如果本地环境缺少 PyTorch，PPO 和 baseline 训练入口会报 `ModuleNotFoundError: No module named 'torch'`。

## Generate Orders

默认一次生成 4 类订单商品数、每类 12 个月，共 48 个算例：

```powershell
python -m data.generate_orders --output-dir data/instances --seed 0
```

输出结构：

```text
data/instances/items_2/orders_m01.pkl  ... orders_m12.pkl
data/instances/items_4/orders_m01.pkl  ... orders_m12.pkl
data/instances/items_6/orders_m01.pkl  ... orders_m12.pkl
data/instances/items_10/orders_m01.pkl ... orders_m12.pkl
```

可筛选生成范围：

```powershell
python -m data.generate_orders --items 2 10 --months 1 12 --output-dir data/instances --seed 0
```

## Train PPO

```powershell
python -m agent.ppo.train --mode short --items 2 --month 1 --episodes 3000 --seed 0
python -m agent.ppo.train --mode long --items 2 --month 1 --episodes 3000 --seed 0
python -m agent.ppo.train --mode hybrid --items 2 --month 1 --episodes 3000 --seed 0
```

常用参数：

```text
--config      配置文件路径，默认 configs/default.json
--mode        short | long | hybrid
--items       单个订单商品数算例，例如 2、4、6、10
--month       月份算例编号，1 到 12
--episodes    训练 episode 数
--seed        随机种子
--output-dir  PPO 输出目录，默认 result/ppo
--max-days    short/hybrid 每个 episode 最大决策天数
--visdom      启用 Visdom
```

## Train Baselines

```powershell
python agent\baselines\a2c.py --mode short --items 2 --month 1 --episodes 10 --seed 0
python agent\baselines\ddpg.py --mode long --items 2 --month 1 --episodes 10 --seed 0
python agent\baselines\td3.py --mode hybrid --items 2 --month 1 --episodes 10 --seed 0
python agent\baselines\sac.py --mode short --items 2 --month 1 --episodes 10 --seed 0
```

## Outputs

- PPO: `result/ppo/<mode>_i<items>_m<month>_seed<seed>.csv`
- PPO checkpoint: `result/ppo/<mode>_i<items>_m<month>_seed<seed>.pth`
- PPO best resource configuration: `result/ppo/<mode>_i<items>_m<month>_seed<seed>_best_config.csv`
- Baselines: `result/baselines/<algorithm>_<mode>_i<items>_m<month>_seed<seed>.csv`
- Baseline best resource configuration: `result/baselines/<algorithm>_<mode>_i<items>_m<month>_seed<seed>_best_config.csv`

训练指标 CSV 包含总成本、延期成本、机器人成本、拣货员成本、订单完成数、准时完成数、完成率、平均拣选时间、订单商品数、月份、训练模式和随机种子。`_best_config.csv` 会在发现更优 `total_cost` 时覆盖更新，保存当前最优 episode 的逐决策机器人和拣货员配置轨迹。

## Tests

```powershell
python -m py_compile environment\class_public.py environment\class_warehouse.py environment\warehouse_env.py data\generate_orders.py agent\ppo\train.py
python -m unittest discover -s test -p "test_*.py"
```

PPO CLI 测试在未安装 PyTorch 时会自动跳过。
