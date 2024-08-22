"""
定义了公共的类
"""

class Order:
    def __init__(self, order_id, items, arrive_time=0):
        self.order_id = order_id  # 订单的编号
        self.items = items  # 订单中的商品列表
        self.arrive_time = arrive_time  # 订单到达时间
        self.complete_time = None  # 订单拣选完成时间
        # 单位拣选时间成本
        self.unit_time_cost = 1
        # 订单中的未拣选完成的商品列表
        self.unpicked_items = items
        # 订单中的已拣选完成的商品列表
        self.picked_items = []