import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from torch import nn

# 1. 生成模拟数据 (与之前相同)
np.random.seed(42)
X_numpy = np.linspace(-2 * np.pi, 2 * np.pi,200).reshape(-1,1)
Y_numpy = np.sin(X_numpy) + 0.05 * np.random.randn(200,1)

x = torch.from_numpy(X_numpy).float()
y = torch.from_numpy(Y_numpy).float()

print("数据生成完成(y = sin(x) + noise)。")
print("---" * 10)

# 2. 直接创建参数张量 a 和 b
class MLP(nn.Module):
    def __init__(self):
        super(MLP,self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(1,64),
            nn.ReLU(),
            nn.Linear(64,64),
            nn.ReLU(),
            nn.Linear(64,1),
        )
    def forward(self,x):
        return self.layers(x)

model = MLP()
print("神经网络模型已创建（3层MLP）。")
print("---" * 10)

# 3. 定义损失函数和优化器
# 损失函数仍然是均方误差 (MSE)。
loss_fn = torch.nn.MSELoss() # 回归任务

# PyTorch 会自动根据这些参数的梯度来更新它们。
optimizer = torch.optim.Adam(model.parameters(),lr=0.01) # 优化器，基于 a b 梯度 自动更新

# 4. 训练模型
num_epochs = 1000
for epoch in range(num_epochs):
    # 前向传播：手动计算 y_pred = a * X + b
    y_pred =model(x)


    # 计算损失
    loss = loss_fn(y_pred, y)

    # 反向传播和优化
    optimizer.zero_grad()  # 清空梯度， torch 梯度 累加
    loss.backward()        # 计算梯度
    optimizer.step()       # 更新参数

    # 每100个 epoch 打印一次损失
    if (epoch + 1) % 1 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 5. 打印最终学到的参数
print("\n训练完成！")
print("---" * 10)

# 6. 绘制结果
# 使用最终学到的参数 a 和 b 来计算拟合直线的 y 值
X_dense = np.linspace(-2 *np.pi, 2 *np.pi,500).reshape(-1,1)
X_dense_tensor = torch.from_numpy(X_dense).float()

model.eval()
with torch.no_grad():
    y_pred_dense = model(X_dense_tensor).numpy()
    y_true_dense = np.sin(X_dense)

plt.figure(figsize=(10, 6))
plt.scatter(X_numpy, Y_numpy, label='Noise data', color='blue', alpha=0.6,s=15)
plt.plot(X_dense,y_true_dense,'--',label='True sin(x)', color='green', linewidth=2)
plt.plot(X_dense,y_pred_dense,label='Neural Net Fit', color='red', linewidth=2)
plt.xlabel('X')
plt.ylabel('y')
plt.title('Fitting sin(x) with a Multilayer Neural Network')
plt.legend()
plt.grid(True)
plt.show()
