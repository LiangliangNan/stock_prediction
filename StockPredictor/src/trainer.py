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
    self.loss_history = []  # 用于记录每轮的平均 Loss

  def train_single_stock(self, symbol, dataloader, model_name_suffix=""):
    """
    model_name_suffix: 用于区分不同日期的模型后缀
    """
    self.model.train()
    self.loss_history = []
    print(f"\n[START] 开始训练股票: {symbol} | 设备: {self.config.DEVICE}")

    for epoch in range(self.config.EPOCHS):
      total_loss = 0
      pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{self.config.EPOCHS}", leave=False)

      for batch_x, batch_y in pbar:
        batch_x = batch_x.to(self.config.DEVICE)
        batch_y = batch_y.to(self.config.DEVICE).unsqueeze(1)

        output = self.model(batch_x)
        loss = self.criterion(output, batch_y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({"Loss": f"{loss.item():.6f}"})

      avg_loss = total_loss / len(dataloader)
      self.loss_history.append(avg_loss)

      if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Epoch [{epoch + 1}/{self.config.EPOCHS}] - Avg Loss: {avg_loss:.6f}")

    # 保存模型和 Loss 曲线
    model_filename = f"{symbol}_{model_name_suffix}_lstm.pth" if model_name_suffix else f"{symbol}_lstm.pth"
    self.save_model(model_filename)
    self.plot_loss(model_filename.replace(".pth", "_loss.png"))

  def plot_loss(self, filename):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(self.loss_history) + 1), self.loss_history, label='Train Loss')
    plt.title('Training Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True)
    save_path = os.path.join(self.config.MODEL_DIR, filename)
    plt.savefig(save_path)
    plt.close()
    print(f"[SUCCESS] Loss 曲线已保存: {save_path}")

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
    else:
      print(f"[WARN] 未找到模型文件: {load_path}")
      return False
