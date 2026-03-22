import torch
import os

class Config:
    # --- 基础路径配置 ---
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    RAW_DATA_DIR = os.path.join(BASE_DIR, "../data/train_data")
    SCALER_DIR = os.path.join(BASE_DIR, "../data/scaler_params")
    MODEL_DIR = os.path.join(BASE_DIR, "../models/checkpoints")
    OUTPUT_DIR = os.path.join(BASE_DIR, "../output")

    # --- 模型选择配置 ---
    # 可选: 'lstm', 'gru', 'attention_lstm', 'transformer'
    MODEL_NAME = 'transformer'

    # --- 特征工程配置 ---
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
    INPUT_DIM = len(FEATURE_COLS)  # 自动根据 FEATURE_COLS 长度计算 (当前为 12)
    HIDDEN_DIM = 128 # 对 transformer，HIDDEN_DIM 必须能被 nhead（多头注意力的头数）整除
    NUM_LAYERS = 2
    OUTPUT_DIM = 1
    DROPOUT = 0.2

    # --- 训练超参数 ---
    BATCH_SIZE = 32
    LR = 0.0005
    EPOCHS = 100
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def init_dirs(cls):
        for d in [cls.SCALER_DIR, cls.MODEL_DIR, cls.OUTPUT_DIR]:
            if not os.path.exists(d):
                os.makedirs(d, exist_ok=True)

Config.init_dirs()
