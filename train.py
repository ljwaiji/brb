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

# 初始化可训练参数，使用更好的初始化方法
# 使用均匀分布初始化theta，避免全部为1的情况
theta = torch.rand(L, requires_grad=True, device=device) * 0.5 + 0.25  # 初始化在[0.25, 0.75]范围内
# 使用均匀分布初始化delta，避免全部为1的情况
delta = torch.rand(T, requires_grad=True, device=device) * 0.5 + 0.25  # 初始化在[0.25, 0.75]范围内
# 使用更为均衡的beta初始化
beta = torch.ones(L, N, device=device) / N  # 均匀初始化beta，使每个后件具有相同的初始可能性
beta = beta + torch.randn(L, N, device=device) * 0.01  # 添加小扰动
beta = beta.clone().detach().requires_grad_(True)  # 设置为可训练

# 创建优化器和学习率调度器
optimizer = Adan(
    [theta, delta, beta],
    lr=1e-3,  # 增大学习率
    betas=(0.98, 0.92, 0.99),  # 使用更激进的动量参数
    eps=1e-8,  # 适度的数值稳定性
    weight_decay=0.01,  # 增加权重衰减以提供更强的正则化
    max_grad_norm=1.0,  # 启用梯度裁剪以稳定训练
    no_prox=False
)
# 替换学习率调度策略
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, 
    T_0=100,  # 初始周期
    T_mult=2,  # 周期倍增因子
    eta_min=1e-5  # 最小学习率
)

# 早停机制设置
early_stopping_patience = 150
early_stopping_counter = 0
best_loss = float('inf')
best_params = None
reg_strength = 0.01  # 初始正则化强度

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
        mse_loss = torch.mean((y_hat.squeeze() - batch_y)**2)
        
        # 添加正则化项，促进参数分布更均衡
        # 1. 添加熵正则项，使beta参数更均衡分布
        entropy_reg = -torch.mean(torch.sum(F.softmax(beta, dim=1) * F.log_softmax(beta, dim=1), dim=1))
        # 2. 为theta添加中心正则项，避免theta全部靠近0或1
        theta_center_reg = torch.mean((theta - 0.5).pow(2))
        # 3. 为delta添加中心正则项，避免delta全部靠近0或1
        delta_center_reg = torch.mean((delta - 0.5).pow(2))
        
        # 组合损失
        loss = mse_loss - reg_strength * entropy_reg + reg_strength * (theta_center_reg + delta_center_reg)
        
        # 使用Adan优化器更新参数
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 确保参数满足各自的约束条件
        with torch.no_grad():
            theta.data.clamp_(0, 1)  # 规则权重约束在[0,1]
            delta.data.clamp_(0, 1)  # 属性权重约束在[0,1]
            beta.data = torch.nn.functional.softmax(beta.data, dim=1)  # beta归一化
        
        epoch_loss += mse_loss.item()  # 仍然记录MSE损失，便于比较
    
    # 更新学习率
    scheduler.step()
    
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
    
    # 获取当前学习率
    current_lr = optimizer.param_groups[0]['lr']
    
    # 监控参数分布情况
    with torch.no_grad():
        theta_avg = theta.mean().item()
        delta_avg = delta.mean().item()
        beta_entropy = -torch.mean(torch.sum(beta * torch.log(beta + 1e-10), dim=1)).item()
    
    print(f'Epoch {epoch+1}/{epochs} | Train Loss: {epoch_loss:.4f} | Test Loss: {test_loss:.4f} | LR: {current_lr:.6f}')
    print(f'Params: theta_avg={theta_avg:.3f}, delta_avg={delta_avg:.3f}, beta_entropy={beta_entropy:.3f}')
    
    # 动态调整正则化强度 - 训练初期较弱，后期较强
    if epoch > epochs // 2:
        # 训练后期增加正则化强度
        reg_strength = min(0.05, 0.01 + (epoch - epochs // 2) * 0.0001)  # 逐渐增加到0.05
    
    # 早停机制
    if test_loss < best_loss:
        best_loss = test_loss
        early_stopping_counter = 0
        best_params = {
            'theta': theta.data.clone(),
            'delta': delta.data.clone(),
            'beta': beta.data.clone()
        }
    else:
        early_stopping_counter += 1
        if early_stopping_counter >= early_stopping_patience:
            print("早停机制触发，训练停止")
            # 恢复最佳模型参数
            theta.data = best_params['theta']
            delta.data = best_params['delta']
            beta.data = best_params['beta']
            break

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