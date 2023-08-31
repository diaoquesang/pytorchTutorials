import random
def split_dataset(file_path, train_ratio=0.8):
    # 读取数据集
    with open(file_path, 'r') as f:
        data = f.readlines()
    # 随机打乱数据集
    random.shuffle(data)
    # 计算训练集和测试集的边界
    train_size = int(len(data) * train_ratio)
    test_size = len(data) - train_size
    # 划分训练集和测试集
    train_set = data[:train_size]
    test_set = data[train_size:]
    # 保存训练集和测试集到对应的txt文件
    with open('0train_set.txt', 'w') as f:
        f.writelines(train_set)
    with open('0test_set.txt', 'w') as f:
        f.writelines(test_set)
    print(f"数据集已成功划分为训练集和测试集，并保存到对应的txt文件中。")
# 调用函数，以8：2的比例划分训练集和测试集
split_dataset('0output.txt',0.8)