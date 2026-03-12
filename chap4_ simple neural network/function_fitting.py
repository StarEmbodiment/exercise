import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# -------------------- 1. 生成数据 --------------------
def target_function(x):
    return np.sin(2*x) + 0.5 * np.cos(3*x) + 0.1 * x**2

N_train, N_test = 500, 200
x_train = np.linspace(-3, 3, N_train).reshape(-1, 1)
x_test = np.linspace(-3, 3, N_test).reshape(-1, 1)

y_train = target_function(x_train) + 0.1 * np.random.randn(N_train, 1)
y_test = target_function(x_test)

# -------------------- 2. 初始化网络参数（单隐藏层） --------------------
input_dim = 1
hidden_dim = 128
output_dim = 1

# He 初始化
W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
b1 = np.zeros((1, hidden_dim))
W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2.0 / hidden_dim)
b2 = np.zeros((1, output_dim))

# -------------------- 3. Adam优化器参数 --------------------
learning_rate = 0.001
beta1, beta2, epsilon = 0.9, 0.999, 1e-8

# 为每个参数初始化一阶矩和二阶矩
m_W1, v_W1 = np.zeros_like(W1), np.zeros_like(W1)
m_b1, v_b1 = np.zeros_like(b1), np.zeros_like(b1)
m_W2, v_W2 = np.zeros_like(W2), np.zeros_like(W2)
m_b2, v_b2 = np.zeros_like(b2), np.zeros_like(b2)

# -------------------- 4. 前向传播（单隐藏层） --------------------
def forward(x, W1, b1, W2, b2):
    z1 = x @ W1 + b1
    a1 = np.maximum(0, z1)
    y_pred = a1 @ W2 + b2
    return y_pred, a1, z1

def compute_loss(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)

# -------------------- 5. 反向传播（单隐藏层） --------------------
def backward(x, y_true, y_pred, a1, z1, W2):
    N = x.shape[0]
    dL_dy = 2 * (y_pred - y_true) / N

    # 输出层梯度
    dL_dW2 = a1.T @ dL_dy
    dL_db2 = np.sum(dL_dy, axis=0, keepdims=True)

    # 隐藏层梯度
    dL_da1 = dL_dy @ W2.T
    dL_dz1 = dL_da1 * (z1 > 0)
    dL_dW1 = x.T @ dL_dz1
    dL_db1 = np.sum(dL_dz1, axis=0, keepdims=True)

    return dL_dW1, dL_db1, dL_dW2, dL_db2

# -------------------- 6. Adam更新规则 --------------------
def adam_update(param, grad, m, v, t, lr, beta1, beta2, epsilon):
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * (grad ** 2)
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)
    param -= lr * m_hat / (np.sqrt(v_hat) + epsilon)
    return param, m, v

# -------------------- 7. 训练 --------------------
epochs = 3000
loss_history = []
t = 0

for epoch in range(epochs):
    y_pred, a1, z1 = forward(x_train, W1, b1, W2, b2)
    loss = compute_loss(y_pred, y_train)
    loss_history.append(loss)

    dW1, db1, dW2, db2 = backward(x_train, y_train, y_pred, a1, z1, W2)

    t += 1
    W1, m_W1, v_W1 = adam_update(W1, dW1, m_W1, v_W1, t, learning_rate, beta1, beta2, epsilon)
    b1, m_b1, v_b1 = adam_update(b1, db1, m_b1, v_b1, t, learning_rate, beta1, beta2, epsilon)
    W2, m_W2, v_W2 = adam_update(W2, dW2, m_W2, v_W2, t, learning_rate, beta1, beta2, epsilon)
    b2, m_b2, v_b2 = adam_update(b2, db2, m_b2, v_b2, t, learning_rate, beta1, beta2, epsilon)

    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Train Loss: {loss:.6f}")

# -------------------- 8. 测试 --------------------
y_test_pred, _, _ = forward(x_test, W1, b1, W2, b2)
test_loss = compute_loss(y_test_pred, y_test)
print(f"Final Test Loss: {test_loss:.6f}")

# -------------------- 9. 可视化 --------------------
plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.plot(loss_history)
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training Loss')

plt.subplot(1,2,2)
plt.scatter(x_train, y_train, s=5, alpha=0.3, label='Train data')
plt.plot(x_test, target_function(x_test), 'g-', linewidth=2, label='True function')
plt.plot(x_test, y_test_pred, 'r--', linewidth=2, label='Prediction')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Function Fitting')

plt.tight_layout()
plt.show()