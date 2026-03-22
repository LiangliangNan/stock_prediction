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

    print(f"\n[START] 开始训练股票: {symbol}, 设备: {self.config.DEVICE}", flush=True)
    print(f"[INFO] 模型输入维度: {len(self.config.FEATURE_COLS)}")
    print(f"[INFO] 特征列表: {self.config.FEATURE_COLS}\n", flush=True)

    total_epochs = self.config.EPOCHS
    width = len(str(total_epochs))

    # 1. 预定义对齐格式
    custom_format = "{desc}: {percentage:3.0f}%|{bar}| {n_fmt:>" + str(
      width) + "}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"

    # 2. 核心优雅改动：leave=False 结合手动收尾
    with tqdm(total=total_epochs, desc=f"TRAIN-{symbol}", unit="ep", dynamic_ncols=True, bar_format=custom_format,
              leave=True) as pbar:
      for epoch in range(total_epochs):
        total_loss = 0
        for batch_x, batch_y in dataloader:
          batch_x, batch_y = batch_x.to(self.config.DEVICE), batch_y.to(self.config.DEVICE).unsqueeze(1)
          output = self.model(batch_x)
          loss = self.criterion(output, batch_y)
          self.optimizer.zero_grad()
          loss.backward()
          self.optimizer.step()
          total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        self.loss_history.append(avg_loss)

        current_ep = epoch + 1
        pbar.set_postfix({"Loss": f"{avg_loss:.6f}"})

        # A. 逻辑解耦：中间过程使用 pbar.write
        if current_ep % 10 == 0 or current_ep == 1:
          if current_ep < total_epochs:
            log_msg = f"  > Epoch [{current_ep:>{width}}/{total_epochs}] - Avg Loss: {avg_loss:8.6f}"
            pbar.write(log_msg)

        pbar.update(1)

      # B. 关键点：在 with 块结束前，通过 pbar.write 强制把最后一行压入历史栈
      # 这样它会被视为“旧日志”，从而被进度条顶上去
      final_msg = f"  > Epoch [{total_epochs:>{width}}/{total_epochs}] - Avg Loss: {self.loss_history[-1]:8.6f}"
      pbar.write(final_msg)

    # 3. 此时进度条已 close() 并停留在 100% 状态。
    # 我们不再在这里写 print，避免产生额外的空行。

    model_filename = f"{symbol}_{model_name_suffix}.pth" if model_name_suffix else f"{symbol}.pth"
    self.save_model(model_filename)
    self.plot_loss(model_filename.replace(".pth", "_loss.png"))

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
