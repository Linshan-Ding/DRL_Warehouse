# Paper Title: The Impact of Dynamic Workforce Allocation in Human-Robot Collaborative Warehouses: A Human-Centric Perspective
## 项目结构

```
DRL_Warehouse/
├── README.md                           # 项目说明文件（本文件）
├── agent/                              # 强化学习算法模块
│   ├── PPO_I.py                        # 短租算法
│   ├── PPO_II.py                       # 长租算法
│   └── PPO_III.py                      # 长短租结合算法
├── data/                               # 数据模块
│   ├── instances                       # .pkl算例存储位置
│   ├── generat_order_lamdas.py         # 算例生成程序
│   └── warehouse.py                    # 仓库环境，以辅助算例生成
├── environment/                        # 环境模块
│   ├── class_public.py                 # 参数定义：仓库参数、机器人参数、拣货员参数、订单参数、商品参数、算法参数
│   ├── class_warehouse.py              # 类定义：订单、商品、储货位、拣货位、机器人和拣货员等类
│   └── warehouse_test2.py              # 仓库环境数值仿真环境
└── results/                            # 结果文件
```
