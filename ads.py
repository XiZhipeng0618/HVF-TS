import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random, pickle
# import torch
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors
from scipy.stats import norm
from datetime import datetime

random.seed(42)
np.random.seed(42)

current_time = datetime.now()
formatted_time = current_time.strftime("%Y-%m-%d %H:%M")

X_train = np.load("./X_train.npy")
X_train_reshaped = X_train.reshape(X_train.shape[0], -1)

table = X_train_reshaped.copy()
print(f"table.shape = :{table.shape}")

# 参数初始化
k = 20
n = table.shape[0]
# 设置随机种子
random_seed = 42
np.random.seed(random_seed)
sampled_data = np.zeros(n, dtype=int)  # 采样数据数组初始化为0
seed_n = 1000
ind = np.random.choice(n, seed_n, replace=False)  # 随机选择 seed_n 个索引
sampled_data[ind] = 1  # 将初始种子样本的位置标记为1
seed = table[ind, :]
density_seed = np.zeros(seed_n)

# initialize loop counter
loop_counter = 0

# 初始化密度种子
# nbrs = NearestNeighbors(n_neighbors=k, algorithm='kd_tree', metric='l1').fit(table)
nbrs = NearestNeighbors(n_neighbors=k).fit(table)
for i in range(seed_n):
    distances, _ = nbrs.kneighbors([seed[i]])
    rk = np.max(distances)
    density_seed[i] = rk

mean_density = np.mean(density_seed)
std_density = np.std(density_seed)

# 初始化记录参数变化列表
loop_counter_record = []
recorded_parameters = []
sampled_indices_record = []

# 设置参数
num = len(X_train_reshaped)
counter = seed_n
max_samples = 700000
save_count = seed_n  # 记录保存的次数
save_interval = 10000  # 每 100,00 样本保存一次
print(f"Max samples: {max_samples}")

# 采样过程
for _ in range(num):
    loop_counter += 1
    ind = np.random.choice(n, 1)[0]
    distances, _ = nbrs.kneighbors([table[ind]])
    rk = np.max(distances)
    
    mean_density_estimate = mean_density + (rk - mean_density) / counter
    std_density_estimate = np.sqrt(((counter - 2) / (counter - 1) * std_density**2) + (1 / counter * (rk - mean_density)**2))
    
    accepted_prob = norm.cdf(rk, mean_density_estimate, std_density_estimate)
    if accepted_prob > np.random.rand():

        mean_density = mean_density_estimate
        std_density = std_density_estimate
        sampled_data[ind] = 1
        counter = int(np.sum(sampled_data)) + 1

        # 记录每100个采样的参数变化
        if counter % 100 == 0:
            loop_counter_record.append((counter, loop_counter))
        recorded_parameters.append((counter, mean_density, std_density))

        # 每当达到保存间隔时，保存采样结果
        if (counter-1) % save_interval == 0:
            sampled_indices = np.where(np.array(sampled_data) == 1)[0]
            
            X_train_sampled = X_train_reshaped[sampled_indices]

            print(X_train_sampled.shape)

            sampled_indices_record.append(sampled_indices)  # 记录当前保存的索引

            # 保存样本数据和相应索引
            save_filename_train_data = f"./X_train_sampled_part.npy"
            save_filename_indices    = f"./sampled_indices_part.npy"

            np.save(save_filename_train_data, X_train_sampled)
            np.save(save_filename_indices, sampled_indices)
            print(f"Saved {save_filename_train_data} with {len(sampled_indices)} sampled indices.")
            
            save_count += 1

        # 达到最大采样数量时退出循环
        if counter > max_samples:
            break

# 获取最终采样的所有数据和索引
sampled_indices = np.where(sampled_data == 1)[0]
X_train_sampled = X_train_reshaped[sampled_indices]


# 保存所有采样的数据、索引、记录的参数
np.save("./X_train_sampled.npy", X_train_sampled)
np.save("./loop_counter_record.npy", loop_counter_record)

