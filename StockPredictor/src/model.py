"""
-------------------------------------------------------------------------
模型	      |  核心优势	                      |    适用场景
----------|---------------------------------|----------------------------
LSTM	    |  经典，具备长短期记忆能力	          |  基础基准测试
GRU	      |  结构简化，收敛速度快，抗过拟合更好	  |  数据量较少、算力有限时
Attention |	不再只看最后一天，能回溯历史关键节点	|  存在明显季节性或突发事件的行情
----------|-------------------------------------------------------------
"""

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


"""
使用 GRU（更现代、参数更少）
GRU 是 LSTM 的变体，它将“遗忘门”和“输入门”合并为一个“更新门”。
优点：参数比 LSTM 少 33%，计算更快，且在小样本（如特定股票数据）上更不容易过拟合。
"""
class StockGRU(nn.Module):
  def __init__(self, config):
    super(StockGRU, self).__init__()
    # 直接将 nn.LSTM 替换为 nn.GRU
    self.rnn = nn.GRU(
      input_size=config.INPUT_DIM,
      hidden_size=config.HIDDEN_DIM,
      num_layers=config.NUM_LAYERS,
      batch_first=True,
      dropout=config.DROPOUT if config.NUM_LAYERS > 1 else 0
    )
    self.fc = nn.Linear(config.HIDDEN_DIM, config.OUTPUT_DIM)

  def forward(self, x):
    # GRU 的输出不包含 cell state (c)，只返回 out 和 hidden state (h)
    out, _h = self.rnn(x)
    prediction = self.fc(out[:, -1, :])
    return prediction

"""
方案二：LSTM + Attention（捕捉关键交易日）
普通的 LSTM 倾向于“最后一天最重要”，但股市中某几天的剧烈波动（如财报日）对未来影响巨大。引入注意力机制可以让模型自动学习应该关注序列中的哪一天。
优点：大幅提升模型对历史波动的敏感度。
替换复杂度：低，只需要在 forward 里加几行矩阵运算。
"""
class StockAttentionLSTM(nn.Module):
  def __init__(self, config):
    super(StockAttentionLSTM, self).__init__()
    self.lstm = nn.LSTM(
      config.INPUT_DIM,
      config.HIDDEN_DIM,
      config.NUM_LAYERS,
      batch_first=True,
      dropout=config.DROPOUT if config.NUM_LAYERS > 1 else 0
    )

    # 注意力权重层
    self.attention = nn.Linear(config.HIDDEN_DIM, 1)
    self.fc = nn.Linear(config.HIDDEN_DIM, config.OUTPUT_DIM)

  def forward(self, x):
    # out shape: [batch, seq_len, hidden_dim]
    out, _ = self.lstm(x)

    # 计算注意力权重 (Score)
    # weights shape: [batch, seq_len, 1]
    weights = torch.softmax(self.attention(out), dim=1)

    # 将权重作用于所有时间步，并求和 (Context Vector)
    # context shape: [batch, hidden_dim]
    context = torch.sum(weights * out, dim=1)

    prediction = self.fc(context)
    return prediction


def get_model(config):
  """
  模型工厂函数：根据 config.MODEL_NAME 自动返回实例化的模型对象
  """
  model_map = {
    "lstm": StockLSTM,
    "gru": StockGRU,
    "attention_lstm": StockAttentionLSTM
  }

  model_key = config.MODEL_NAME.lower()

  if model_key not in model_map:
    print(f"[WARNING] 未定义的模型类型 '{model_key}'，默认返回 LSTM")
    model_class = StockLSTM
  else:
    model_class = model_map[model_key]
    print(f"[INFO] 当前使用模型架构: {model_key.upper()}")

  # 实例化并移动到指定设备 (CPU/CUDA)
  return model_class(config).to(config.DEVICE)
