import torch
import numpy as np
from sklearn.model_selection import train_test_split
from adan import Adan
import utils as ut
import ebrb_torch as et

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载数据
data = np.loadtxt('C:\\Users\\liu2021\\Desktop\\brb_test\\oil.data')
X = torch.tensor(data[:, :2], dtype=torch.float32, device=device)  # 输入特征：压力和流量
y = torch.tensor(data[:, 2], dtype=torch.float32, device=device)   # 标签：泄漏状态

# 数据标准化
X = (X - X.mean(dim=0)) / X.std(dim=0)
y = (y - y.mean()) / y.std()

# 划分训练集和测试集（8:2比例）
X_train, X_test, y_train, y_test = train_test_split(X.cpu().numpy(), y.cpu().numpy(), test_size=0.2, random_state=42)
X_train = torch.tensor(X_train, device=device)
X_test = torch.tensor(X_test, device=device)
y_train = torch.tensor(y_train, device=device)
y_test = torch.tensor(y_test, device=device)

# 设置BRB参数
reference_num = 7
L = 56  # 规则数
N = 5   # 后件参考值个数（Z, VS, M, H, VH）
T = 2   # 前件属性个数（FlowDiff和PressureDiff）

# 创建BRB模型
brb = ut.create_brb(X_train, y_train, reference_num, N)

# 初始化可训练参数（根据论文参数定义）
theta = torch.ones(L, requires_grad=True, device=device)     # 规则权重
delta = torch.ones(T, requires_grad=True, device=device)     # 属性权重
beta = torch.randn(L, N, requires_grad=True, device=device)  # 置信度参数

# 创建优化器和学习率调度器
optimizer = Adan([theta, delta, beta], lr=1e-3, betas=(0.98, 0.92, 0.99))
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
        batch_X = X_train[indices]
        batch_y = y_train[indices]
        
        # 转换输入格式
        brb_input = et.transform_input(batch_X, brb, batch_X.shape[0])
        
        # 前向传播
        infer = et.Inference(
            M=batch_size,
            L=L,
            N=N,
            T=T,
            antecedent_mask_r=brb.antecedent_reference_value,
            brb_input=brb_input,
            theta=theta,
            delta=delta,
            beta=beta,
            antecedent_reference_value=brb.antecedent_reference_value,
            consequent_reference=brb.consequent_reference,
            task='regression'
        )
        outputs = infer.execute()
        
        # 计算损失
        loss = torch.mean((outputs.squeeze() - batch_y)**2)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 参数约束
        with torch.no_grad():
            theta.data.clamp_(0, 1)  # 规则权重约束在[0,1]
            delta.data.clamp_(0, 1)  # 属性权重约束在[0,1]
            beta.data = torch.nn.functional.softmax(beta.data, dim=1)  # beta归一化
        
        epoch_loss += loss.item()
    
    # 更新学习率
    scheduler.step(epoch_loss)
    
    # 验证
    with torch.no_grad():
        test_input = et.transform_input(X_test, brb, X_test.shape[0])
        test_infer = et.Inference(
            M=X_test.shape[0],
            L=L,
            N=N,
            T=T,
            antecedent_mask_r=brb.antecedent_reference_value,
            brb_input=test_input,
            theta=theta,
            delta=delta,
            beta=beta,
            antecedent_reference_value=brb.antecedent_reference_value,
            consequent_reference=brb.consequent_reference,
            task='regression'
        )
        test_outputs = test_infer.execute()
        test_loss = torch.mean((test_outputs.squeeze() - y_test)**2)
    
    print(f'Epoch {epoch+1}/{epochs} | Train Loss: {epoch_loss:.4f} | Test Loss: {test_loss:.4f}')

# 最终测试
with torch.no_grad():
    final_input = et.transform_input(X_test, brb, X_test.shape[0])
    final_infer = et.Inference(
        M=X_test.shape[0],
        L=L,
        N=N,
        T=T,
        antecedent_mask_r=brb.antecedent_reference_value,
        brb_input=final_input,
        theta=theta,
        delta=delta,
        beta=beta,
        antecedent_reference_value=brb.antecedent_reference_value,
        consequent_reference=brb.consequent_reference,
        task='regression'
    )
    final_outputs = final_infer.execute()
    final_loss = torch.mean((final_outputs.squeeze() - y_test)**2)
    print(f'Final Test Loss: {final_loss:.4f}')