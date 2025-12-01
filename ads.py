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

# 将项目根目录添加到系统路径
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')# 获取当前时间

current_time = datetime.now()
formatted_time = current_time.strftime("%Y-%m-%d %H:%M")

# 加载训练集
df_train=pd.read_csv('data-tmp/datas/train_datas.csv')

print("len(df) = ", len(df_train))

H = df_train['high'].values
L = df_train['low'].values
df_train['vol_parkinson'] = np.sqrt((1 / (4 * np.log(2))) * (np.log(H / L) ** 2))

#############################################################
# 滑动窗口划分
def create_sequences(data, seq_len=48, step=24):
    X, y = [], []
    
    for i in range(len(data) - seq_len - step):
        input_data = data[i:i+seq_len]
        values = input_data
        future_price = data[i + seq_len:i + seq_len + step]  # 取 close 列
        # 保存数据
        X.append(values)
        y.append(future_price)
    
    return np.array(X), np.array(y)

# 使用 DataFrame 传递所有特征
# X, y = create_sequences(df_train[['close']].values)
X, y = create_sequences(df_train[['vol_parkinson']].values)

# # 使用 DataFrame 传递所有特征
# X, y = create_sequences(df_train[['close', 'volume']].values)

#############################################################
# 计算样本和标签对应平均值及标准差
x_mean=X.mean(axis=(0, 1))
x_std=X.std(axis=(0, 1))

y_mean=y.mean(axis=(0, 1))
y_std=y.std(axis=(0, 1))

# 标准化处理
X=(X-x_mean)/x_std
y=(y-y_mean)/y_std

data=[]
for i in range(len(X)):
    data.append((X[i],y[i]))

# 将数据保存到文件中
with open('/root/data-tmp/VLDB/Sampling_Ori/saved_data/Mean_Std_Waves.pkl', 'wb') as f:
    pickle.dump({'x_mean': x_mean, 'x_std': x_std, 'y_mean': y_mean, 'y_std': y_std}, f)

print("数据已保存到 Mean_Std_Waves.pkl 文件中")
###############################################################################

# 打乱数据集，提高模型泛化性
# random.seed(42)  # 设置随机种子
random.shuffle(data)  # 现在每次打乱的顺序都是相同的

train_ratio = 0.8
train_size = int(len(data) * train_ratio)

train_data = data[:train_size]
validation_data = data[train_size:]

# 将数据拆分为训练集和验证集的输入和输出部分
X_train, y_train = zip(*train_data)
X_val, y_val = zip(*validation_data)

# 转换为 numpy 数组
X_train, y_train = np.array(X_train), np.array(y_train)
X_val, y_val = np.array(X_val), np.array(y_val)
# Display the sizes of each set
print(f"Train set size: {len(train_data), X_train.shape}")
print(f"Validation set size: {len(validation_data), X_val.shape}")

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
            y_train_sampled = y_train[sampled_indices]

            print(X_train_sampled.shape, y_train_sampled.shape)

            sampled_indices_record.append(sampled_indices)  # 记录当前保存的索引

            # 保存样本数据和相应索引
            save_filename_train_data = f"/root/data-tmp/VLDB/Sampling_Ori/saved_data/X_train_sampled_part_{len(sampled_indices)}.npy"
            save_filename_indices    = f"/root/data-tmp/VLDB/Sampling_Ori/saved_data/sampled_indices_part_{len(sampled_indices)}.npy"
            save_filename_label_data = f"/root/data-tmp/VLDB/Sampling_Ori/saved_data/y_train_sampled_part_{len(sampled_indices)}.npy"

            np.save(save_filename_train_data, X_train_sampled)
            np.save(save_filename_label_data, y_train_sampled)
            np.save(save_filename_indices, sampled_indices)
            print(f"Saved {save_filename_train_data} with {len(sampled_indices)} sampled indices.")
            
            save_count += 1

        # 达到最大采样数量时退出循环
        if counter > max_samples:
            break

# 获取最终采样的所有数据和索引
sampled_indices = np.where(sampled_data == 1)[0]
y_train_sampled = y_train[sampled_indices]
X_train_sampled = X_train_reshaped[sampled_indices]
print("Number of sampled indices:", len(sampled_indices))
print("X_train_sampled.shape:", X_train_sampled.shape)

# 保存所有采样的数据、索引、记录的参数
np.save("/root/data-tmp/VLDB/Sampling_Ori/saved_data/X_train_sampled.npy", X_train_sampled)
np.save("/root/data-tmp/VLDB/Sampling_Ori/saved_data/y_train_sampled.npy", y_train_sampled)
np.save("/root/data-tmp/VLDB/Sampling_Ori/saved_data/loop_counter_record.npy", loop_counter_record)
np.save("/root/data-tmp/VLDB/Sampling_Ori/saved_data/recorded_parameters.npy", recorded_parameters)
