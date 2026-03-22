import torch
import os


class Config:
  # --- 基础路径配置 ---
  BASE_DIR = os.path.dirname(os.path.abspath(__file__))
  RAW_DATA_DIR = os.path.join(BASE_DIR, "../data/train_data")
  SCALER_DIR = os.path.join(BASE_DIR, "../data/scaler_params")
  MODEL_DIR = os.path.join(BASE_DIR, "../models/checkpoints")
  OUTPUT_DIR = os.path.join(BASE_DIR, "../output")

  # --- 特征工程配置 (匹配 9 列 CSV) ---
  FEATURE_COLS = [
    'open', 'high', 'low', 'close', 'vol',  # 基础特征
    'ma5', 'ma20', 'v_ma5',                 # 均线特征
    'bias_5', 'v_ratio',                    # 衍生特征
    'returns', 'amplitude'                  # 动量特征
  ]
  TARGET_COL = 'close'

  # --- 时间序列参数 ---
  SEQ_LEN = 60
  PREDICT_STEP = 1

  # --- 模型架构参数 ---
  INPUT_DIM = len(FEATURE_COLS)  # 此时为 8
  HIDDEN_DIM = 128 # HIDDEN_DIM = 64 对于 8 个特征可能略显单薄
  NUM_LAYERS = 2
  OUTPUT_DIM = 1
  DROPOUT = 0.2

  # --- 训练超参数 ---
  BATCH_SIZE = 32
  LR = 0.0005   # 如果 Loss 剧烈抖动就调小，如果不动就调大
  EPOCHS = 100   # 100
  DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  @classmethod
  def init_dirs(cls):
    for d in [cls.SCALER_DIR, cls.MODEL_DIR, cls.OUTPUT_DIR]:
      if not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


Config.init_dirs()
