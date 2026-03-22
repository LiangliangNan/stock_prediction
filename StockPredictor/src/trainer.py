import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import matplotlib.pyplot as plt


class Trainer:
  def __init__(self, config, model):
    self.config = config
    self.model = model.to(config.DEVICE)
    self.criterion = nn.MSELoss()
    self.optimizer = optim.Adam(self.model.parameters(), lr=config.LR)
    self.loss_history = []

  def train_single_stock(self, symbol, dataloader, model_name_suffix=""):
    self.model.train()
    self.loss_history = []
    print(f"\n[START] 开始训练股票: {symbol}, 设备: {self.config.DEVICE}")

    total_epochs = self.config.EPOCHS
    width = len(str(total_epochs))  # 计算总轮数的数字宽度

    for epoch in range(total_epochs):
      total_loss = 0
      # 使用变量 width 动态对齐 desc
      pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1:>{width}}/{total_epochs}", leave=False)

      for batch_x, batch_y in pbar:
        batch_x, batch_y = batch_x.to(self.config.DEVICE), batch_y.to(self.config.DEVICE).unsqueeze(1)
        output = self.model(batch_x)
        loss = self.criterion(output, batch_y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({"Loss": f"{loss.item():.6f}"})

      avg_loss = total_loss / len(dataloader)
      self.loss_history.append(avg_loss)

      # 严格对齐输出逻辑
      if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == total_epochs - 1:
        log_msg = f"\tEpoch [{epoch + 1:>{width}}/{total_epochs}] - Avg Loss: {avg_loss:8.6f}"
        tqdm.write(log_msg)

    # 1. 确定模型和图片名称
    model_filename = f"{symbol}_{model_name_suffix}.pth" if model_name_suffix else f"{symbol}.pth"
    loss_img_name = model_filename.replace(".pth", "_loss.png")

    # 2. 保存模型到 MODEL_DIR
    self.save_model(model_filename)

    # 3. 保存 Loss 曲线到 OUTPUT_DIR (路径对齐改进)
    self.plot_loss(loss_img_name)

  def plot_loss(self, filename):
    plt.figure(figsize=(10, 5))
    epochs = range(1, len(self.loss_history) + 1)
    plt.plot(epochs, self.loss_history, label='Train Loss', color='blue')

    # --- 新增：在曲线末端标注最后的 Loss 数值 ---
    last_epoch = epochs[-1]
    last_loss = self.loss_history[-1]

    plt.annotate(f'Final Loss: {last_loss:.6f}',
                 xy=(last_epoch, last_loss),
                 xytext=(-10, 10),  # 文字相对于点的偏移：向左10像素，向上10像素
                 textcoords='offset points',
                 ha='right',  # 水平对齐：右对齐，防止文字超出右边界
                 va='bottom',  # 垂直对齐：底部对齐
                 fontsize=10,
                 color='red',
                 fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color='red', alpha=0.5))
    # ----------------------------------------

    plt.title('Training Loss Convergence', fontsize=12)
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)

    # 核心改动：保存到 MODEL_DIR
    save_path = os.path.join(self.config.MODEL_DIR, filename)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[SUCCESS] Loss 曲线已导出至: {save_path}")

  def save_model(self, filename):
    save_path = os.path.join(self.config.MODEL_DIR, filename)
    torch.save(self.model.state_dict(), save_path)
    print(f"[SUCCESS] 模型权重已保存: {save_path}")

  def load_model(self, filename):
    load_path = os.path.join(self.config.MODEL_DIR, filename)
    if os.path.exists(load_path):
      self.model.load_state_dict(torch.load(load_path, map_location=self.config.DEVICE))
      self.model.eval()
      print(f"[LOG] 加载模型成功: {load_path}")
      return True
    return False
