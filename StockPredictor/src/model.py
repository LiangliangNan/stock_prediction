import torch
import torch.nn as nn


class StockLSTM(nn.Module):
  def __init__(self, config):
    super(StockLSTM, self).__init__()
    self.config = config

    # LSTM 层
    # input_size: 特征维度 (9), hidden_size: 隐藏层神经元 (64)
    self.lstm = nn.LSTM(
      input_size=config.INPUT_DIM,
      hidden_size=config.HIDDEN_DIM,
      num_layers=config.NUM_LAYERS,
      batch_first=True,  # 输入形状为 [batch, seq, feature]
      dropout=config.DROPOUT if config.NUM_LAYERS > 1 else 0
    )

    # 全连接输出层
    # 将 LSTM 最后一个时间步的输出映射到 1 个预测值 (收盘价)
    self.fc = nn.Linear(config.HIDDEN_DIM, config.OUTPUT_DIM)

  def forward(self, x):
    # x shape: [batch, seq_len, input_dim]

    # out shape: [batch, seq_len, hidden_dim]
    # _h, _c 分别是最后一个时间步的隐藏状态和细胞状态
    out, (_h, _c) = self.lstm(x)

    # 我们只需要序列中最后一个时间步的输出来做预测
    # out[:, -1, :] 取出 [batch, 1, hidden_dim]
    last_step_out = out[:, -1, :]

    # 得到预测结果 [batch, 1]
    prediction = self.fc(last_step_out)
    return prediction
