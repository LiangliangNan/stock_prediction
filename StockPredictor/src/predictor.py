import torch
import pandas as pd
import numpy as np
import joblib
import os
from datetime import timedelta


class Predictor:
  def __init__(self, config, model):
    self.config = config
    self.model = model.to(config.DEVICE)
    self.model.eval()

  def _update_indicators(self, df):
    df['ma5'] = df['close'].rolling(window=5).mean()
    df['ma20'] = df['close'].rolling(window=20).mean()
    df['v_ma5'] = df['vol'].rolling(window=5).mean()
    df = df.bfill().ffill()
    return df

  def predict_future(self, symbol, days=5):
    scaler_path = os.path.join(self.config.SCALER_DIR, f"{symbol}_scaler.pkl")
    if not os.path.exists(scaler_path):
      raise FileNotFoundError(f"找不到 {symbol} 的归一化参数。")
    scaler = joblib.load(scaler_path)

    file_path = os.path.join(self.config.RAW_DATA_DIR, f"{symbol}.csv")
    # 同步 9 列读取
    df = pd.read_csv(file_path, header=0, names=[
      'ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'change', 'pct_chg', 'vol'
    ])

    df['trade_date'] = pd.to_datetime(df['trade_date'].astype(str))
    current_df = df.tail(self.config.SEQ_LEN + 20).copy()
    predictions = []

    with torch.no_grad():
      for i in range(days):
        temp_df = self._update_indicators(current_df)
        last_features = temp_df[self.config.FEATURE_COLS].tail(self.config.SEQ_LEN).values
        scaled_input = scaler.transform(last_features)
        input_tensor = torch.FloatTensor(scaled_input).unsqueeze(0).to(self.config.DEVICE)

        pred_normalized = self.model(input_tensor)

        # 反归一化 (特征数已改为 8)
        dummy_row = np.zeros((1, len(self.config.FEATURE_COLS)))
        target_idx = self.config.FEATURE_COLS.index(self.config.TARGET_COL)
        dummy_row[0, target_idx] = pred_normalized.item()
        real_pred_price = scaler.inverse_transform(dummy_row)[0, target_idx]

        last_date = current_df['trade_date'].iloc[-1]
        next_date = last_date + timedelta(days=1)

        new_row = {
          'ts_code': symbol, 'trade_date': next_date,
          'close': real_pred_price, 'open': real_pred_price,
          'high': real_pred_price, 'low': real_pred_price,
          'vol': current_df['vol'].mean()  # 移除 amount
        }

        current_df = pd.concat([current_df, pd.DataFrame([new_row])], ignore_index=True)
        predictions.append({
          'date': next_date.strftime('%Y-%m-%d'),
          'predicted_close': round(real_pred_price, 2)
        })
    return pd.DataFrame(predictions)
