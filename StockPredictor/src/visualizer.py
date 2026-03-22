"""
================================================================================
MODULE: Stock Prediction Visualizer
DESCRIPTION:
    - 检查并加载训练好的模型权重。
    - 调用 Predictor 预测未来 N 天的走势。
    - 读取历史最后 30 天的数据。
    - 将历史实线与预测虚线平滑对齐，生成可视化报告。
================================================================================
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import torch
import os
import sys # 新增：用于优雅退出程序
from config import Config
from model import StockLSTM
from predictor import Predictor


def plot_prediction(symbol, future_days=10):
  cfg = Config()

  # 1. 检查模型文件是否存在 (逻辑增强)
  model_filename = f"{symbol}_lstm.pth"
  model_path = os.path.join(cfg.MODEL_DIR, model_filename)

  if not os.path.exists(model_path):
      print(f"\n" + "!"*60)
      print(f"[ERROR] 找不到模型权重文件: {model_path}")
      print(f"[TIPS]  请先运行训练脚本 (如 train.py) 为股票 {symbol} 训练模型。")
      print(f"[TIPS]  或者检查 config.py 中的 MODEL_DIR 路径设置是否正确。")
      print("!"*60 + "\n")
      return # 退出当前函数

  # 2. 加载模型
  try:
      model = StockLSTM(cfg)
      model.load_state_dict(torch.load(model_path, map_location=cfg.DEVICE))
      model.eval() # 确保进入评估模式
      print(f"[INFO] 成功加载模型权重: {model_filename}")
  except Exception as e:
      print(f"[ERROR] 模型加载失败: {e}")
      return

  # 3. 获取预测数据
  pd_engine = Predictor(cfg, model)
  future_df = pd_engine.predict_future(symbol, days=future_days)

  # 4. 读取历史数据用于对比 (取最后30天)
  file_path = os.path.join(cfg.RAW_DATA_DIR, f"{symbol}.csv")
  if not os.path.exists(file_path):
      print(f"[ERROR] 找不到历史数据文件: {file_path}")
      return

  history_df = pd.read_csv(file_path, header=0, names=[
    'ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'change', 'pct_chg', 'vol'
  ])
  history_df['trade_date'] = pd.to_datetime(history_df['trade_date'].astype(str))
  recent_history = history_df.tail(30)

  # 5. 绘图
  plt.figure(figsize=(12, 6))

  # 画历史线 (蓝色)
  plt.plot(recent_history['trade_date'], recent_history['close'], label='History', color='blue', marker='o')

  # 画预测线 (对齐逻辑)
  last_hist_date = recent_history['trade_date'].iloc[-1]
  last_hist_close = recent_history['close'].iloc[-1]

  pred_dates = [last_hist_date] + pd.to_datetime(future_df['date']).tolist()
  pred_prices = [last_hist_close] + future_df['predicted_close'].tolist()

  # 绘制预测虚线
  plt.plot(pred_dates, pred_prices, label='Prediction', color='red', linestyle='--',
           marker='s', markevery=range(1, len(pred_dates)))

  # 标注最新价格点
  label_text = f"{last_hist_date.strftime('%Y-%m-%d')}\nLatest: {last_hist_close:.2f}"
  plt.annotate(label_text,
               xy=(last_hist_date, last_hist_close),
               xytext=(10, 15),
               textcoords='offset points',
               arrowprops=dict(arrowstyle='->', color='green'))

  plt.title(f"Stock {symbol} Price Prediction (Next {future_days} Days)")
  plt.xlabel("Date")
  plt.ylabel("Price")

  # --- 新增：网格加密逻辑 ------------------------------------------------------------------------
  ax = plt.gca()
  # 主要刻度：每5天显示一次文字
  ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
  ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
  # 次要刻度：每天一个网格点
  ax.xaxis.set_minor_locator(mdates.DayLocator(interval=1))
  # 绘制加密网格
  ax.grid(which='major', linestyle='-', linewidth='0.5', color='gray', alpha=0.5)
  ax.grid(which='minor', linestyle=':', linewidth='0.3', color='silver', alpha=0.8)
  # --------------------------------------------------------------------------------------------

  plt.legend(loc='upper left') # 响应你的习惯，放到左侧
  plt.grid(True, linestyle=':', alpha=0.6)
  plt.xticks(rotation=45)
  plt.tight_layout()

  # 保存
  output_filename = f"{symbol}_future_forecast.png"
  output_path = os.path.join(cfg.OUTPUT_DIR, output_filename)
  plt.savefig(output_path, dpi=300)
  print(f"\n[SUCCESS] 预测图表已保存至: {output_path}")

  plt.show()


if __name__ == "__main__":
  plot_prediction("600879", future_days=5)
