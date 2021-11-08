# 01_fruits.py
# 图像分类：水果分类
# 数据集介绍：
# 1307张水果图片，共5个类别（苹果288张，香蕉275张，葡萄216张，橘子277张，梨子251张）

# 01.数据预处理
import os
import json

name_dict = {'apple': 0, 'banana': 1, 'grape': 2, 'orange': 3, 'pear': 4}  # 名称-分类数字对应字典
data_root_path = 'fruits/'  # 数据集目录
test_file_path = data_root_path + 'test.list'  # 测试集文件路径
train_file_path = data_root_path + 'train.list'  # 训练集文件路径
readme_file = data_root_path + 'readme.json'  # 样本数据汇总文件
name_data_list = {}  # 记录每个类别多少张训练图片、测试图片


def save_train_test_file(path, name):
    if name not in name_data_list:
        img_list = []
        img_list.append(path)  # 将图片添加到列表
        name_data_list[name] = img_list  # 将图片列表存入字典
    else:  # 某类水果已经在字典中
        name_data_list[name].append(path)  # 直接加入


# 遍历目录、将图片路径存入字典、再由字典写入文件
dirs = os.listdir(data_root_path)  # 列出fruits、目录下所有内容
for d in dirs:
    full_path = data_root_path + d  # 拼完整目录路径
    if os.path.isdir(full_path):  # 如果是目录，遍历目录中的图片
        imgs = os.listdir(full_path)
        for img in imgs:
            save_train_test_file(full_path + '/' + img, d)
        else:  # 如果是文件，不作处理
            pass

# 分测试集和训练集
with open(test_file_path, 'w') as f:
    pass
with open(train_file_path, 'w')as f:
    pass
# 遍历字典，每10个数据分1个到测试集
for name, img_list in name_data_list.items():
    i = 0
    num = len(img_list)  # 打印每一张图片的张数
    print(f'{name}:{num}张')
    for img in img_list:
        if i % 10 == 0:  # 放入测试集
            with open(test_file_path, 'a') as f:
                line = f'{img}\t{name_dict[name]}\n'  # 拼一行
                f.write(line)  # 写入
        else:  # 放入训练集
            with open(train_file_path, 'a') as f:
                line = f'{img}\t{name_dict[name]}\n'  # 拼一行
                f.write(line)  # 写入
        i += 1

# 02.网络搭建、模型训练、保存
import paddle
import paddle.fluid as fluid
import cv2
import numpy as np
import sys
from multiprocessing import cpu_count
import matplotlib.pyplot as plt

paddle.enable_static()


def train_mapper(sample):
    img, label = sample  # sample由图路径、标机组成
    if not os.path.exists(img):
        print('图片不存在：', img)
    else:
        # 读取图片，并对图片做维度变化
        img = paddle.dataset.image.load_image(img)  # 读取图像
        # 对图片进行变换，修剪，输出（3,100,100）的矩阵
        img = paddle.dataset.image.simple_transform(im=img,
                                                    resize_size=100,
                                                    crop_size=100,
                                                    is_color=True,
                                                    is_train=True)
        # 图像归一化处理，将值压缩到0~1之间
        img = img.flatten().astype('float32') / 255.0
        return img, label


# 自定义reader，从训练集读取数据，并交给train_mapper处理
def train_r(train_list, buffered_size=1024):
    def reader():
        with open(train_list, 'r') as f:
            lines = [line.strip() for line in f]
            for line in lines:
                img_path, lab = line.strip().split("\t")
                yield img_path, int(lab)

    return paddle.reader.xmap_readers(train_mapper,  # mapper函数
                                      reader,  # reader
                                      cpu_count(),  # 线程数
                                      buffered_size)  # 缓冲区大小


# 搭建神经网络
# 输入层 --> 卷积-池化层/dropout --> 卷积池化层/dropout --> 卷积池化层/dropout
# --> 全连接层 --> dropout --> 全连接层
def convolution_nural_network(image, type_size):
    # 第一个卷积-池化层
    conv_pool_1 = fluid.nets.simple_img_conv_pool(
        input=image,  # 输入数据
        filter_size=3,  # 卷积核大小
        num_filters=32,  # 卷积核大小，与输出通道数相同
        pool_size=2,  # 池化层2*2
        pool_stride=2,  # 池化层步长
        act='relu')  # 激活函数
    # dropout:丢弃学习，随机丢弃一些神经元的输出，防止过拟合
    drop = fluid.layers.dropout(x=conv_pool_1,  # 输入
                                dropout_prob=0.5)  # 丢弃率

    # 第二个卷积-池化层
    conv_pool_2 = fluid.nets.simple_img_conv_pool(
        input=drop,  # 输入数据
        filter_size=3,  # 卷积核大小
        num_filters=64,  # 卷积核大小，与输出通道数相同
        pool_size=2,  # 池化层2*2
        pool_stride=2,  # 池化层步长
        act='relu')  # 激活函数
    drop = fluid.layers.dropout(x=conv_pool_2,  # 输入
                                dropout_prob=0.5)  # 丢弃率

    # 第三个卷积-池化层
    conv_pool_3 = fluid.nets.simple_img_conv_pool(
        input=drop,  # 输入数据
        filter_size=3,  # 卷积核大小
        num_filters=64,  # 卷积核大小，与输出通道数相同
        pool_size=2,  # 池化层2*2
        pool_stride=2,  # 池化层步长
        act='relu')  # 激活函数
    drop = fluid.layers.dropout(x=conv_pool_3,  # 输入
                                dropout_prob=0.5)  # 丢弃率

    # 全连接层
    fc = fluid.layers.fc(input=drop, size=512, act='relu')
    drop = fluid.layers.dropout(x=fc, dropout_prob=0.5)
    predict = fluid.layers.fc(input=drop,  # 输入层
                              size=type_size,  # 最终分类个数
                              act='softmax')  # 激活函数
    return predict


# 准备数据执行训练
BATCH_SIZE = 32
trainer_reader = train_r(train_list=train_file_path)
train_reader = paddle.batch(paddle.reader.shuffle(reader=trainer_reader,
                                                  buf_size=1200),
                            batch_size=BATCH_SIZE)
# 训练时的输入数据
image = fluid.layers.data(name='image',
                          shape=[3, 100, 100],  # RBG三通道彩色图像
                          dtype='float32')
# 训练时期望的输出值（真实类别）
label = fluid.layers.data(name='label',
                          shape=[1],
                          dtype='int64')
# 调用函数，创建卷积神经网络
predict = convolution_nural_network(image=image,  # 输入数据
                                    type_size=5)  # 类别数量
cost = fluid.layers.cross_entropy(input=predict,  # 预测值
                                  label=label)  # 期望值
avg_cost = fluid.layers.mean(cost)  # 求瞬时值的平均值
# 计算预测准确率
accuracy = fluid.layers.accuracy(input=predict,  # 预测值
                                 label=label)  # 期望值
# 定义优化器（自适应梯度下降）
optimizer = fluid.optimizer.Adam(learning_rate=0.001)
optimizer.minimize(avg_cost)  # 损失值的最小值优化
# 定义执行器
place = fluid.CUDAPlace(0)  # GPU上执行
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())  # 初始化系统参数
feeder = fluid.DataFeeder(feed_list=[image, label],
                          place=place)  # 数据喂入
for pass_id in range(80):
    train_cost = 0
    for batch_id, data in enumerate(train_reader()):
        train_cost, train_acc = exe.run(
            program=fluid.default_main_program(),  # 执行默认program
            feed=feeder.feed(data),  # 输入数据
            fetch_list=[avg_cost, accuracy])  # 获取结果
        if batch_id % 20 == 0:  # 每20次训练打印一笔
            print(f'pass:{pass_id}, batch:{batch_id}, cost:{train_cost[0]}, acc:{train_acc[0]}')
print('训练完成！')

# 保存模型
model_save_dir = 'fruits/'
if not os.path.exists(model_save_dir):
    os.mkdir(model_save_dir)
fluid.io.save_inference_model(dirname=model_save_dir,
                              feeded_var_names=['image'],  # 投喂数据类型
                              target_vars=[predict],  # 给出结果类型
                              executor=exe)
print('保存模型完成！')