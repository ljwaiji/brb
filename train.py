import torch
import numpy as np
from sklearn.model_selection import train_test_split
from adan import Adan
import utils as ut
import ebrb_torch as et
from torch.nn import functional as F

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载数据
data = np.loadtxt('oil.data')
X = torch.tensor(data[:, :2], dtype=torch.float32, device=device)  # 输入特征：压力和流量
y = torch.tensor(data[:, 2], dtype=torch.float32, device=device)   # 标签：泄漏状态

# 数据标准化 - 使用Min-Max标准化代替Z-Score标准化
X_min, X_max = X.min(dim=0)[0], X.max(dim=0)[0]
X = (X - X_min) / (X_max - X_min)
y_min, y_max = y.min(), y.max()
y = (y - y_min) / (y_max - y_min)

# 设置BRB参数
reference_num = 7
N = 5   # 后件参考值个数
T = 2   # 前件属性个数
L = reference_num ** T

# 创建BRB模型
brb = ut.create_brb(X, y, reference_num, N)

# 创建BRB输入
brb_X = et.transform_input(X, brb, X.shape[0])

# 划分训练集和测试集（8:2比例）
train_index, test_index = train_test_split(np.arange(X.shape[0]), test_size=0.2, random_state=42)

# 分割原始特征和标签
X_train = X[train_index].to(device)
X_test = X[test_index].to(device)
y_train = y[train_index].to(device)
y_test = y[test_index].to(device)

# 分割BRB格式的输入
brb_X_train = brb_X[train_index].to(device)
brb_X_test = brb_X[test_index].to(device)

# 初始化可训练参数
theta = torch.ones(L, requires_grad=True, device=device)     # 规则权重
delta = torch.ones(T, requires_grad=True, device=device)     # 属性权重
beta = torch.randn(L, N, requires_grad=True, device=device)  # 置信度参数

# 创建优化器和学习率调度器
optimizer = Adan(
    [theta, delta, beta],
    lr=1e-3,  # 增大学习率
    betas=(0.98, 0.92, 0.99),  # 使用更激进的动量参数
    eps=1e-8,  # 适度的数值稳定性
    weight_decay=0.0,  # 适当的正则化强度
    max_grad_norm=0.0,  # 放宽梯度裁剪
    no_prox=False
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=50, factor=0.5)

# 训练参数
epochs = 1000
batch_size = 32
train_size = X_train.shape[0]

# 训练循环
for epoch in range(epochs):
    permutation = torch.randperm(train_size)
    epoch_loss = 0.0
    
    for i in range(0, train_size, batch_size):
        indices = permutation[i:i+batch_size]
        batch_y = y_train[indices]
        
        # 使用已转换的BRB输入
        batch_brb_X = brb_X_train[indices]
        
        # 按照用户要求创建推理对象
        infer = et.Inference(
            M=batch_brb_X.shape[0],
            L=L,
            N=N,
            T=T,
            antecedent_mask_r=brb.antecedent_reference_value,
            brb_input=batch_brb_X,  # 使用BRB格式的输入
            theta=theta,
            delta=delta,
            beta=beta,
            antecedent_reference_value=brb.antecedent_reference_value,
            consequent_reference=brb.consequent_reference,
            task='regression'
        )
        
        # 执行推理获取预测值
        y_hat = infer.execute()
        
        # 计算MSE损失
        loss = torch.mean((y_hat.squeeze() - batch_y)**2)
        
        # 使用Adan优化器更新参数
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 确保参数满足各自的约束条件
        with torch.no_grad():
            theta.data.clamp_(0, 1)  # 规则权重约束在[0,1]
            delta.data.clamp_(0, 1)  # 属性权重约束在[0,1]
            beta.data = torch.nn.functional.softmax(beta.data, dim=1)  # beta归一化
        
        epoch_loss += loss.item()
    
    # 更新学习率
    scheduler.step(epoch_loss)
    
    # 验证
    with torch.no_grad():
        # 使用训练好的参数对测试集进行预测
        test_infer = et.Inference(
            M=brb_X_test.shape[0],
            L=L,
            N=N,
            T=T,
            antecedent_mask_r=brb.antecedent_reference_value,
            brb_input=brb_X_test,  # 使用测试集的BRB格式输入
            theta=theta,
            delta=delta,
            beta=beta,
            antecedent_reference_value=brb.antecedent_reference_value,
            consequent_reference=brb.consequent_reference,
            task='regression'
        )
        # 执行推理获取测试集预测值
        y_test_hat = test_infer.execute()
        # 计算测试集MSE损失
        test_loss = torch.mean((y_test_hat.squeeze() - y_test)**2)
    
    print(f'Epoch {epoch+1}/{epochs} | Train Loss: {epoch_loss:.4f} | Test Loss: {test_loss:.4f}')

# 最终测试
with torch.no_grad():
    # 使用训练好的参数对测试集进行最终预测
    final_infer = et.Inference(
        M=brb_X_test.shape[0],
        L=L,
        N=N,
        T=T,
        antecedent_mask_r=brb.antecedent_reference_value,
        brb_input=brb_X_test,  # 使用测试集的BRB格式输入
        theta=theta,
        delta=delta,
        beta=beta,
        antecedent_reference_value=brb.antecedent_reference_value,
        consequent_reference=brb.consequent_reference,
        task='regression'
    )
    # 执行推理获取最终测试集预测值
    y_test_hat = final_infer.execute()
    # 计算最终测试集MSE损失
    final_loss = torch.mean((y_test_hat.squeeze() - y_test)**2)
    print(f'Final Test Loss: {final_loss:.4f}')
    print(f'训练完成，最终参数：')
    print(f'theta: {theta.data}')
    print(f'delta: {delta.data}')
    print(f'beta shape: {beta.shape}')
    print(f'前5个规则的beta值: {beta[:5]}')