# import torch
#
# # 检查是否有可用的 CUDA 设备
# if torch.cuda.is_available():
#   print(f"CUDA is available! Number of GPUs: {torch.cuda.device_count()}")
#
#   # 显示每个 CUDA 设备的名称
#   for i in range(torch.cuda.device_count()):
#     print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
# else:
#   print("CUDA is not available")

import random
import time

def fair_lottery(participants, num_winners, allow_duplicates=False, weights=None, seed=None):
    """
    模拟一个更公平的抽签系统。

    参数:
    participants: 参与者列表，例如 ['Alice', 'Bob', 'Charlie'] 或 [{'id': 1, 'name': 'Alice'}, {'id': 2, 'name': 'Bob'}]
    num_winners:  中奖人数
    allow_duplicates: 是否允许同一参与者多次中奖 (默认为 False，更公平)
    weights: (可选) 参与者的权重列表，用于实现加权抽签，例如 [1, 2, 1] 表示 Bob 的权重是 Alice 和 Charlie 的两倍
    seed: (可选) 随机数种子，用于复现抽签结果

    返回值:
    中奖者列表 (如果参与者是字典，则返回字典列表；否则返回名字列表)
    """

    if not participants:
        return []  # 如果没有参与者，返回空列表

    if num_winners <= 0:
        return [] # 如果不需要中奖者，返回空列表

    if seed is not None:
        random.seed(seed) # 设置随机数种子，保证可复现性

    population = [] # 存放所有抽签单位的列表

    if weights is None:
        # 默认情况：每个参与者一个抽签单位
        population = participants
    else:
        # 加权抽签：根据权重，每个参与者可能有多个抽签单位
        if len(weights) != len(participants):
            raise ValueError("权重列表的长度必须与参与者列表的长度一致")
        for i, participant in enumerate(participants):
            weight = weights[i]
            if weight <= 0:
                raise ValueError("权重必须为正数")
            population.extend([participant] * weight) # 根据权重添加抽签单位

    if num_winners >= len(population) and not allow_duplicates:
        if allow_duplicates:
            winners = random.choices(population, k=num_winners) # 允许重复中奖时，直接用 choices
        else:
            winners = population[:] # 如果中奖人数大于等于总人数且不允许重复，则所有人都是赢家
            random.shuffle(winners) # 打乱顺序，确保随机性
            winners = winners[:num_winners] # 截取前 num_winners 个
    else:
        if allow_duplicates:
            winners = random.choices(population, k=num_winners) # 允许重复中奖时，用 choices
        else:
            winners = random.sample(population, k=num_winners) # 不允许重复中奖时，用 sample

    return winners


# 示例用法

# # 1. 基本抽签，每个参与者机会均等，不允许重复中奖
# participants_basic = ['Alice', 'Bob', 'Charlie', 'David', 'Eve']
# num_winners_basic = 2
# winners_basic = fair_lottery(participants_basic, num_winners_basic)
# print(f"基本抽签 - 参与者: {participants_basic}, 中奖人数: {num_winners_basic}, 中奖者: {winners_basic}")
#
#
# # 2. 加权抽签，Bob 的权重是别人的两倍
# participants_weighted = ['Alice', 'Bob', 'Charlie']
# weights_weighted = [1, 1, 1] # Bob 的权重是 2
# num_winners_weighted = 1
# winners_weighted = fair_lottery(participants_weighted, num_winners_weighted, weights=weights_weighted)
# print(f"加权抽签 - 参与者: {participants_weighted}, 权重: {weights_weighted}, 中奖人数: {num_winners_weighted}, 中奖者: {winners_weighted}")
#
#
# # 3. 允许重复中奖的抽签
# participants_duplicate = ['Alice', 'Bob', 'Charlie']
# num_winners_duplicate = 2
# winners_duplicate = fair_lottery(participants_duplicate, num_winners_duplicate, allow_duplicates=True)
# print(f"允许重复中奖抽签 - 参与者: {participants_duplicate}, 中奖人数: {num_winners_duplicate}, 中奖者: {winners_duplicate}")
#
#
# # 4. 使用字典作为参与者，方便携带更多信息
# participants_dict = [
#     {'id': 1, 'name': 'Alice'},
#     {'id': 2, 'name': 'Bob'},
#     {'id': 3, 'name': 'Charlie'},
#     {'id': 4, 'name': 'David'},
#     {'id': 5, 'name': 'Eve'}
# ]
# num_winners_dict = 2
# winners_dict = fair_lottery(participants_dict, num_winners_dict)
# print(f"字典参与者抽签 - 参与者: {participants_dict}, 中奖人数: {num_winners_dict}, 中奖者: {winners_dict}")

# 5. 使用随机数种子，复现抽签结

participants_seed = ['xiaomi', 'jige','None', 'leishen', 'huoying']
num_winners_seed = 1
seed_value = time.time()
winners_seed_1 = fair_lottery(participants_seed, num_winners_seed, seed=seed_value)
# winners_seed_2 = fair_lottery(participants_seed, num_winners_seed, seed=seed_value) # 使用相同的种子
print(f"使用种子抽签 - 参与者: {participants_seed}, 中奖人数: {num_winners_seed}, 种子: {seed_value}, 中奖者 1: {winners_seed_1}")
# print(f"使用种子抽签 - 参与者: {participants_seed}, 中奖人数: {num_winners_seed}, 种子: {seed_value}, 中奖者 2: {winners_seed_2} (结果相同，可复现)")

# # 6. 中奖人数大于等于参与人数，不允许重复中奖
# participants_many_winners = ['Alice', 'Bob']
# num_winners_many_winners = 3
# winners_many_winners = fair_lottery(participants_many_winners, num_winners_many_winners)
# print(f"多中奖者抽签 - 参与者: {participants_many_winners}, 中奖人数: {num_winners_many_winners}, 中奖者: {winners_many_winners}")
#
# # 7. 中奖人数大于等于参与人数，允许重复中奖
# participants_many_winners_duplicate = ['Alice', 'Bob']
# num_winners_many_winners_duplicate = 3
# winners_many_winners_duplicate = fair_lottery(participants_many_winners_duplicate, num_winners_many_winners_duplicate, allow_duplicates=True)
# print(f"多中奖者重复抽签 - 参与者: {participants_many_winners_duplicate}, 中奖人数: {num_winners_many_winners_duplicate}, 中奖者: {winners_many_winners_duplicate}")