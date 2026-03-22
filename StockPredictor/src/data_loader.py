"""
================================================================================
MODULE: Data Management & Feature Engineering Pipeline
DESCRIPTION:
    本模块负责从原始数据到深度学习张量的全流程转换。
    采用了“特征工厂”设计模式，支持高度可扩展的特征工程。

DESIGN PHILOSOPHY (设计思路):
    1. 结构化解耦 (Decoupling): 
       将特征计算 (Indicator Calculation) 与数据加载 (Data Loading) 分离。
       通过 `_extract_features` 作为统一入口，像流水线一样串联不同的特征组。
    2. 鲁棒性 (Robustness): 
       统一处理 NaN 值（使用 bfill/ffill），并在除法运算中加入 epsilon 防止除零异常。
    3. 实验友好性 (Researcher Friendly): 
       用户只需在“特征工厂”区新增函数，并在管道中注册即可，无需改动核心预处理逻辑。

USAGE GUIDE (后续测试新特征的方法):
    步骤 1: 在 [FEATURE FACTORY] 区域定义你的新特征函数，例如 `_add_volatility_indicators(self, df)`。
    步骤 2: 在 `_extract_features` 管道函数中添加一行 `df = self._your_new_func(df)`。
    步骤 3: 在 `config.py` 的 `FEATURE_COLS` 列表中同步添加新生成的列名。
    步骤 4: 重新运行训练逻辑（由于使用了 StandardScaler，新特征会自动被归一化）。

DATE: 2026-03-22
AUTHOR: Gemini AI Assistant
================================================================================
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import joblib
import os


class StockDataset(Dataset):
  """封装 PyTorch 数据集"""

  def __init__(self, X, y):
    self.X = torch.FloatTensor(X)
    self.y = torch.FloatTensor(y)

  def __len__(self):
    return len(self.y)

  def __getitem__(self, idx):
    return self.X[idx], self.y[idx]


class DataManager:
  def __init__(self, config):
    self.config = config

  # ============================================================================
  # [SECTION] 特征工厂 (FEATURE FACTORY)
  # 提示：在这里定义新的技术指标计算逻辑
  # ============================================================================

  def _add_price_indicators(self, df):
    """基础趋势与均线类特征"""
    df['ma5'] = df['close'].rolling(window=5).mean()
    df['ma20'] = df['close'].rolling(window=20).mean()
    # 乖离率 (Bias): 衡量股价偏离均线的程度，用于捕捉超买超卖后的反转
    df['bias_5'] = (df['close'] - df['ma5']) / (df['ma5'] + 1e-9)
    return df

  def _add_volume_indicators(self, df):
    """成交量动力学特征"""
    df['v_ma5'] = df['vol'].rolling(window=5).mean()
    # 量比 (Volume Ratio): 衡量相对于均值的成交爆发力
    df['v_ratio'] = df['vol'] / (df['v_ma5'] + 1e-9)
    return df

  def _add_momentum_indicators(self, df):
    """动量与博弈类特征"""
    # 收益率特征 (Returns)
    df['returns'] = df['close'].pct_change()
    # 日内振幅 (Amplitude): 衡量多空博弈的激烈程度
    df['amplitude'] = (df['high'] - df['low']) / (df['close'].shift(1) + 1e-9)
    return df

  def _extract_features(self, df):
    """
    [PIPELINE] 特征管道入口
    如需测试新特征，请在此处注册新的函数调用
    """
    df = df.copy()

    # --- 管道开始 ---
    df = self._add_price_indicators(df)
    df = self._add_volume_indicators(df)
    df = self._add_momentum_indicators(df)
    # --- 管道结束 ---

    # 统一处理缺失值 (由 rolling/shift 产生的空值)
    df = df.bfill().ffill()
    return df

  # ============================================================================
  # [SECTION] 核心预处理流程
  # ============================================================================

  def load_and_preprocess(self, symbol, mode='train', train_end_date=None):
    """
    加载并预处理数据
    symbol: 股票代码
    mode: 'train' (拟合归一化) 或 'test' (沿用归一化)
    train_end_date: 数据截断点 'YYYYMMDD'
    """
    file_path = os.path.join(self.config.RAW_DATA_DIR, f"{symbol}.csv")
    if not os.path.exists(file_path):
      raise FileNotFoundError(f"未找到数据: {file_path}")

    # 1. 原始读取
    df = pd.read_csv(file_path, header=0, names=[
      'ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'change', 'pct_chg', 'vol'
    ])

    # 2. 日期对齐与过滤
    df['trade_date_dt'] = pd.to_datetime(df['trade_date'].astype(str))
    if train_end_date is not None:
      cut_date = pd.to_datetime(train_end_date)
      if mode == 'train':
        df = df[df['trade_date_dt'] <= cut_date].copy()

    # 3. 数据类型强制转换与基础清洗
    numeric_cols = ['open', 'high', 'low', 'close', 'vol']
    for col in numeric_cols:
      df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['close']).reset_index(drop=True)

    # 4. 执行特征管道
    df = self._extract_features(df)

    # 5. 特征提取与索引定位
    feature_data = df[self.config.FEATURE_COLS].values
    target_idx = self.config.FEATURE_COLS.index(self.config.TARGET_COL)

    # 6. 归一化逻辑
    scaler_path = os.path.join(self.config.SCALER_DIR, f"{symbol}_scaler.pkl")
    if mode == 'train':
      scaler = StandardScaler()
      scaled_features = scaler.fit_transform(feature_data)
      joblib.dump(scaler, scaler_path)
    else:
      scaler = joblib.load(scaler_path)
      scaled_features = scaler.transform(feature_data)

    # 7. 构建滚动窗口数据集 (X: [Batch, Seq_Len, Features])
    X, y = [], []
    for i in range(len(scaled_features) - self.config.SEQ_LEN):
      X.append(scaled_features[i: i + self.config.SEQ_LEN])
      y.append(scaled_features[i + self.config.SEQ_LEN, target_idx])

    return np.array(X), np.array(y), scaler, df

  def get_dataloader(self, symbol, train_end_date=None):
    """
    封装 DataLoader 供训练使用
    """
    X, y, _, _ = self.load_and_preprocess(symbol, mode='train', train_end_date=train_end_date)
    dataset = StockDataset(X, y)
    return DataLoader(dataset, batch_size=self.config.BATCH_SIZE, shuffle=True, drop_last=True)
