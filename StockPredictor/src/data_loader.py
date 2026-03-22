import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import joblib
import os


class StockDataset(Dataset):
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

  def _calculate_indicators(self, df):
    df = df.copy()
    df['ma5'] = df['close'].rolling(window=5).mean()
    df['ma20'] = df['close'].rolling(window=20).mean()
    df['v_ma5'] = df['vol'].rolling(window=5).mean()
    df = df.bfill().ffill()
    return df

  def load_and_preprocess(self, symbol, mode='train', train_end_date=None):
    """
    symbol: 股票代码
    mode: 'train' 或 'test'
    train_end_date: 字符串 'YYYYMMDD'，例如 '20231231'
    """
    file_path = os.path.join(self.config.RAW_DATA_DIR, f"{symbol}.csv")
    if not os.path.exists(file_path):
      raise FileNotFoundError(f"未找到数据: {file_path}")

    df = pd.read_csv(file_path, header=0, names=[
      'ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'change', 'pct_chg', 'vol'
    ])

    # 转换日期格式以便过滤
    df['trade_date_dt'] = pd.to_datetime(df['trade_date'].astype(str))

    # 核心逻辑：根据日期截断数据
    if train_end_date is not None:
      cut_date = pd.to_datetime(train_end_date)
      if mode == 'train':
        # 训练模式：只取截止日期之前的数据
        df = df[df['trade_date_dt'] <= cut_date].copy()
      else:
        # 测试/回测模式：保留全量数据，但在外部逻辑中控制起始点
        pass

    numeric_cols = ['open', 'high', 'low', 'close', 'vol']
    for col in numeric_cols:
      df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['close']).reset_index(drop=True)

    df = self._calculate_indicators(df)
    feature_data = df[self.config.FEATURE_COLS].values
    target_idx = self.config.FEATURE_COLS.index(self.config.TARGET_COL)

    scaler_path = os.path.join(self.config.SCALER_DIR, f"{symbol}_scaler.pkl")
    if mode == 'train':
      scaler = StandardScaler()
      scaled_features = scaler.fit_transform(feature_data)
      joblib.dump(scaler, scaler_path)
    else:
      scaler = joblib.load(scaler_path)
      scaled_features = scaler.transform(feature_data)

    X, y = [], []
    for i in range(len(scaled_features) - self.config.SEQ_LEN):
      X.append(scaled_features[i: i + self.config.SEQ_LEN])
      y.append(scaled_features[i + self.config.SEQ_LEN, target_idx])

    return np.array(X), np.array(y), scaler, df

  def get_dataloader(self, symbol, train_end_date=None):
    # 增加 train_end_date 传递
    X, y, _, _ = self.load_and_preprocess(symbol, mode='train', train_end_date=train_end_date)
    dataset = StockDataset(X, y)
    return DataLoader(dataset, batch_size=self.config.BATCH_SIZE, shuffle=True, drop_last=True)
