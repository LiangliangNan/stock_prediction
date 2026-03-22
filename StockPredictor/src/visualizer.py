"""
visualizer.py。
作用是：
  - 调用 Predictor 预测未来 N 天的走势。
  - 读取历史最后 30 天的数据。
  - 把历史曲线和预测曲线连起来画在一张图上，让你直观感受趋势。
"""

import matplotlib.pyplot as plt
import pandas as pd
import torch
import os
from config import Config
from model import StockLSTM
from predictor import Predictor


def plot_prediction(symbol, future_days=10):
  cfg = Config()

  # 1. 加载模型
  model = StockLSTM(cfg)
  model.load_state_dict(torch.load(os.path.join(cfg.MODEL_DIR, f"{symbol}_lstm.pth"), map_location=cfg.DEVICE))

  # 2. 获取预测数据
  pd_engine = Predictor(cfg, model)
  future_df = pd_engine.predict_future(symbol, days=future_days)

  # 3. 读取历史数据用于对比 (取最后30天)
  file_path = os.path.join(cfg.RAW_DATA_DIR, f"{symbol}.csv")
  history_df = pd.read_csv(file_path, header=0, names=[
    'ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'change', 'pct_chg', 'vol'
  ])
  history_df['trade_date'] = pd.to_datetime(history_df['trade_date'].astype(str))
  recent_history = history_df.tail(30)

  # 4. 绘图
  plt.figure(figsize=(12, 6))

  # 画历史线
  plt.plot(recent_history['trade_date'], recent_history['close'], label='History', color='blue', marker='o')

  # 画预测线 (为了连贯，把历史最后一天加进预测线的开头)
  last_hist_date = recent_history['trade_date'].iloc[-1]
  last_hist_close = recent_history['close'].iloc[-1]

  pred_dates = [last_hist_date] + pd.to_datetime(future_df['date']).tolist()
  pred_prices = [last_hist_close] + future_df['predicted_close'].tolist()

  plt.plot(pred_dates, pred_prices, label='Prediction', color='red', linestyle='--', marker='s')

  plt.title(f"Stock {symbol} Price Prediction (Next {future_days} Days)")
  plt.xlabel("Date")
  plt.ylabel("Price")
  plt.legend()
  plt.grid(True)
  plt.xticks(rotation=45)
  plt.tight_layout()

  # 保存到指定的 OUTPUT_DIR
  output_filename = f"{symbol}_future_forecast.png"
  output_path = os.path.join(cfg.OUTPUT_DIR, output_filename)
  plt.savefig(output_path, dpi=300)  # 提高分辨率
  print(f"\n[SUCCESS] 预测图表已保存至: {output_path}")

  # 如果在 GUI 环境下则显示图表
  plt.show()


if __name__ == "__main__":
  plot_prediction("600879", future_days=5)
