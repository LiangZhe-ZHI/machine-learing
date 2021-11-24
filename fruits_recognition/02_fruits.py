# 03.模型加载、执行预测
import numpy as np
import paddle
from PIL import Image
from matplotlib import pyplot as plt
from paddle import fluid

paddle.enable_static()

name_dict = {'apple': 0, 'banana': 1, 'grape': 2, 'orange': 3, 'pear': 4}
model_save_dir = 'fruits/'

place = fluid.CPUPlace()  # 预测不需要在GPU上执行
infer_exe = fluid.Executor(place)  # 执行器


def load_image(path):  # 读取图片，调整尺寸，归一化处理
    img = paddle.dataset.image.load_and_transform(path, 100, 100, False).astype("float32")  # 加载图像
    img = img / 255.0  # 归一化，将像素值压缩到0~1
    return img


infer_imgs = []  # 图像数据列表
test_img = 'C:\\Users\\zlz\\Desktop\\OIP-C.jpg'  # 预测图像路径
infer_imgs.append(load_image(test_img))  # 加载图像数据，添加到列表
infer_imgs = np.array(infer_imgs)  # 转换array！！

# 加载模型
[infer_program, feed_target_names, fetch_targets] = fluid.io.load_inference_model(model_save_dir, infer_exe)
# 先显示原始图像
img = Image.open(test_img)  # 打开图片
plt.imshow(img)
plt.show()
# 执行预测
results = infer_exe.run(infer_program,
                        feed={feed_target_names[0]: infer_imgs},
                        fetch_list=fetch_targets)
print(results)  # result为数组，包含每个类别的概率
result = np.argmax(results[0])  # 获取最大索引值
for k, v in name_dict.items():
    if result == v:
        print('预测结果:', k)