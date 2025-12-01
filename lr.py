import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random, csv, logging
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge, TheilSenRegressor, PassiveAggressiveRegressor, OrthogonalMatchingPursuit
from sklearn.linear_model import LinearRegression, ElasticNet, HuberRegressor, Lars, Lasso, Ridge, TheilSenRegressor, BayesianRidge




random.seed(42)
np.random.seed(42)
# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.set_visible_devices(physical_devices[4], 'GPU')
#############################################################
# 加载训练集
df_train=pd.read_csv('/media/chenjunjie/cdb0f1df-df9d-4704-ade1-ed1fde64603c/xzp/finaces/24_10_25/data/train_datas.csv')

print("len(df) = ", len(df_train))
#############################################################
# 滑动窗口划分
def create_sequences(data, seq_len=30, step=5):
    X, y = [], []
    
    for i in range(len(data) - seq_len - step-seq_len):
        # 提取各特征数据
        input_data = data[i:i+seq_len]
        values = input_data
        # future_price = data[i + seq_len:i + seq_len + step]  # 取 close 列
        future_price = data[i + seq_len:i + seq_len + seq_len]  # 取 close 列
        # 保存数据
        X.append(values)
        y.append(future_price)
    
    return np.array(X), np.array(y)

#  划分样本和对应标签
X, y = create_sequences(df_train[['close', 'volume']].values)
# X, y = create_sequences(df_train[['open', 'high', 'low', 'close', 'volume']].values)

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

# 打乱数据集，提高模型泛化性
random.shuffle(data)

# 训练集和验证集划分
train_ratio = 0.8
train_size = int(len(data) * train_ratio)

train_data = data[:train_size]

# Display the sizes of each set
print(f"Train set size: {len(train_data)}")
# breakpoint()

# 将数据拆分为训练集和验证集的输入和输出部分
X_train, y_train = zip(*train_data)
# X_val, y_val = zip(*validation_data)

# 转换为 numpy 数组
X_train, y_train = np.array(X_train), np.array(y_train)
print('X_train.shape, y_train.shape', X_train.shape, y_train.shape)

# Train the model
print('X_train.shape, y_train.shape', X_train.shape, y_train.shape)
# breakpoint()

print("train.shape = ", X_train.shape, y_train.shape)

#############################################################
# loading training model

# 测试，加载测试数据集
df_test=pd.read_csv('/media/chenjunjie/cdb0f1df-df9d-4704-ade1-ed1fde64603c/xzp/finaces/24_10_25/data/testdata.csv')

# 获取测试样本和相对应标签
X_test, y_test = create_sequences(df_test[['Close', 'Volume']].values)
print(f"X_test.shape, y_test.shape :{X_test.shape, y_test.shape}")

# 测试样本和相对应标签标准化处理
X_test=(X_test-x_mean)/x_std
y_test=(y_test-y_mean)/y_std

# X_test = X_test.reshape(X_test.shape[0], X_test.shape[1] * X_test.shape[2])
# y_test = y_test.reshape(y_test.shape[0], y_test.shape[1] * y_test.shape[2])

# 模型列表
models = [
    ("LinearRegression", LinearRegression()),
    ("BayesianRidge", BayesianRidge()),
    ("Ridge", Ridge()),
    ("TheilSenRegressor", TheilSenRegressor()),
    ("PassiveAggressiveRegressor", PassiveAggressiveRegressor()),
    ("OrthogonalMatchingPursuit", OrthogonalMatchingPursuit()),
    ("ElasticNet", ElasticNet()),
    ("Lars", Lars())
]

# 循环遍历每个模型
for model_name, model in models:
    print(f"Training and evaluating {model_name}...")

    # 初始化预测和目标列表
    predictions = []
    targets = []
    
    # 设置 k 和训练数据
    k = 60
    print(f"X_test.shape, y_test.shape :{X_test.shape, y_test.shape}")
    Remainder = len(X_test) % k

    # 遍历每个样本进行预测
    for i in range(k, len(X_test) - Remainder,k):
        # 对于后续样本，只使用前 k 个样本
        X_train_i = X_train[i - k:i]  # 仅使用前 k 个样本
        y_train_i = y_train[i - k:i]  # 仅使用前 k 个标签

        # 训练模型
        model.fit(X_train_i.reshape(-1, 1), y_train_i.flatten())

        # 进行预测
        pred = model.predict(X_test[i - k:i].reshape(-1, 1))
        predictions.append(pred)  # 存储预测结果
        targets.append(y_test[i - k:i])  # 存储真实结果

    # 将预测结果转换为数组
    y_pred = np.array(predictions)
    targets = np.array(targets)

    # 打印预测结果形状
    print(f"y_pred.shape = :{y_pred.shape, targets.shape}")
    np.save(f"/media/chenjunjie/cdb0f1df-df9d-4704-ade1-ed1fde64603c/xzp/finaces/24_10_25/24-12-20/Linear-Reg/lr/saved_files/{model_name}_predictions.npy", y_pred)

    y_pred_reshape = y_pred.reshape(-1, 1)
    targets_reshape = targets.reshape(-1, 1)
    y_pred_reshape = np.squeeze(y_pred_reshape)
    targets_reshape = np.squeeze(targets_reshape)

    # 计算相关系数
    correlation = np.corrcoef(y_pred_reshape, targets_reshape)[0, 1]
    print(f"Correlation for {model_name}: {correlation}")

    # 绘制图形
    plt.figure(figsize=(15, 8))
    plt.plot(targets_reshape, label=f'{model_name} Actual Bitcoin Closing Log Returns', color='blue', alpha=0.6)
    plt.plot(y_pred_reshape, label=f'{model_name} Predicted Bitcoin Closing Log Returns', color='red', alpha=0.6)
    plt.title(f"{model_name} Predicted vs Actual Bitcoin Closing Log Returns", fontsize=22)
    plt.xlabel('Time (min)', fontsize=18)
    plt.ylabel('Lr Bitcoin Closing Log Returns', fontsize=18)
    plt.legend(loc="best", fontsize=12)
    plt.savefig(f'/media/chenjunjie/cdb0f1df-df9d-4704-ade1-ed1fde64603c/xzp/finaces/24_10_25/24-12-20/Linear-Reg/lr/saved_files/{model_name}_pred_test.jpg', bbox_inches='tight')
    plt.show()

    y_pred = y_pred.reshape(-1, y.shape[1], y.shape[2])
    targets = targets.reshape(-1, y.shape[1], y.shape[2])
    # 计算 MAE, MAPE, RMSE
    pre = y_pred * y_std + y_mean
    gt = targets * y_std + y_mean

    pre = pre.reshape(-1, 1)
    gt = gt.reshape(-1, 1)
    pre = np.squeeze(pre)
    gt = np.squeeze(gt)

    ind = gt != 0

    # 计算 MAPE 和 MAE
    mape = np.mean(np.abs((pre[ind] - gt[ind]) / np.maximum(np.abs(pre[ind]), np.abs(gt[ind])))) * 100
    log_mae = np.mean(np.abs(y_pred.flatten() - targets.flatten())) * 100

    # 计算 RMSE 和 R²
    relative_error = (pre[ind] - gt[ind]) / np.maximum(np.abs(pre[ind]), np.abs(gt[ind]))
    rmse = np.sqrt(np.mean(relative_error ** 2)) * 100

    # R² 计算
    ss_res = np.sum(relative_error ** 2)
    ss_tot = np.sum(((gt[ind] - np.mean(gt[ind])) / np.maximum(np.abs(pre[ind]), np.abs(gt[ind]))) ** 2)
    r2_squre = (1 - ss_res / ss_tot) * 100

    print(f"R² for {model_name}: {r2_squre:.4f}%")
    print(f"Log Mean Absolute Error for {model_name}: {log_mae:.4f}%")
    print(f"Root Mean Squared Error for {model_name}: {rmse:.4f}%")
    print(f"Mean Absolute Percent Error for {model_name}: {mape:.4f}%")
    
    # 记录日志
    model_name_log = f'{model_name}_k={k}'
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    log_files = [
        '/media/chenjunjie/cdb0f1df-df9d-4704-ade1-ed1fde64603c/xzp/finaces/24_10_25/24-12-20/Linear-Reg/lr/saved_files/results.log'
    ]
    for log_file in log_files:
        handler = logging.FileHandler(log_file)
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(handler)

    new_data = {
        'model_name': model_name_log,
        'R2(%)': round(r2_squre, 4),
        'Log MAE(%)': round(log_mae, 4),
        'RMSE(%)': round(rmse, 4),
        'MAPE(%)': round(mape, 4)
    }

    logger.info(f"Model Name: {model_name_log}, R²: {new_data['R2(%)']:.4f}%, "
                f"Log MAE: {new_data['Log MAE(%)']:.4f}%, "
                f"RMSE: {new_data['RMSE(%)']:.4f}%, "
                f"MAPE: {new_data['MAPE(%)']:.4f}%")

    # 将新数据写入 CSV 文件
    csv_files = [
        '/media/chenjunjie/cdb0f1df-df9d-4704-ade1-ed1fde64603c/xzp/finaces/24_10_25/24-12-20/Linear-Reg/lr/saved_files/metrics.csv'
    ]
    for file_path in csv_files:
        with open(file_path, 'a', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['model_name', 'R2(%)', 'Log MAE(%)', 'RMSE(%)', 'MAPE(%)']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            if csvfile.tell() == 0:
                writer.writeheader()

            writer.writerow(new_data)

    print(f"Information written to CSV file: {', '.join(csv_files)}")


