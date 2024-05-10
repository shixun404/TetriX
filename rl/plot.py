import pickle
import matplotlib.pyplot as plt

# 从文件中读取数据
with open('list.txt', 'rb') as file:
    loss_list = pickle.load(file)
for i in range(len(loss_list)):
    loss_list[i] = loss_list[i].detach().cpu()
x= []
y = []
for i in range(0, len(loss_list), 50):
    x.append(i)
    y.append(loss_list[i])
# 绘制损失曲线
plt.figure(figsize=(10, 5))  # 设置图形的尺寸
plt.plot(x, y, label='Loss')  # 绘制损失曲线
plt.title('Loss Curve lambda_2 / lambda_1')  # 添加标题
plt.xlabel('Epochs')  # X轴标签
plt.ylabel('Loss')  # Y轴标签
plt.legend()  # 添加图例
plt.grid(True)  # 显示网格
plt.savefig('loss.png')  # 显示图形