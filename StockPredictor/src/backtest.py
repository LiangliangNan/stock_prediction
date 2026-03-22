import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib
from config import Config
from data_loader import DataManager
from model import StockLSTM
from trainer import Trainer


def run_backtest(symbol, train_end_date, test_days=20):
  cfg = Config()
  mgr = DataManager(cfg)

  model_suffix = f"enddate_{train_end_date}"
  model_filename = f"{symbol}_{model_suffix}_lstm.pth"

  # 1. 训练
  train_loader = mgr.get_dataloader(symbol, train_end_date=train_end_date)
  model = StockLSTM(cfg)
  trainer = Trainer(cfg, model)
  trainer.train_single_stock(symbol, train_loader, model_name_suffix=model_suffix)

  # 2. 回测推理准备
  file_path = os.path.join(cfg.RAW_DATA_DIR, f"{symbol}.csv")
  full_df = pd.read_csv(file_path, header=0,
                        names=['ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'change', 'pct_chg', 'vol'])
  full_df['trade_date_dt'] = pd.to_datetime(full_df['trade_date'].astype(str))
  full_df = full_df.sort_values('trade_date_dt').reset_index(drop=True)

  split_idx = full_df[full_df['trade_date_dt'] <= pd.to_datetime(train_end_date)].index[-1]
  real_future = full_df.iloc[split_idx + 1: split_idx + 1 + test_days].copy()

  # 滚动预测逻辑
  current_df = full_df.iloc[split_idx - cfg.SEQ_LEN - 20: split_idx + 1].copy()
  scaler = joblib.load(os.path.join(cfg.SCALER_DIR, f"{symbol}_scaler.pkl"))
  predictions = []

  with torch.no_grad():
    for _ in range(len(real_future)):
      current_df['ma5'] = current_df['close'].rolling(5).mean()
      current_df['ma20'] = current_df['close'].rolling(20).mean()
      current_df['v_ma5'] = current_df['vol'].rolling(5).mean()
      temp_df = current_df.bfill().ffill()

      last_seq = temp_df[cfg.FEATURE_COLS].tail(cfg.SEQ_LEN).values
      scaled_input = scaler.transform(last_seq)
      input_tensor = torch.FloatTensor(scaled_input).unsqueeze(0).to(cfg.DEVICE)

      pred_norm = model(input_tensor)
      dummy = np.zeros((1, len(cfg.FEATURE_COLS)))
      idx = cfg.FEATURE_COLS.index(cfg.TARGET_COL)
      dummy[0, idx] = pred_norm.item()
      pred_price = scaler.inverse_transform(dummy)[0, idx]

      new_row = {'trade_date_dt': current_df['trade_date_dt'].iloc[-1] + pd.Timedelta(days=1),
                 'close': pred_price, 'open': pred_price, 'high': pred_price, 'low': pred_price,
                 'vol': current_df['vol'].mean()}
      current_df = pd.concat([current_df, pd.DataFrame([new_row])], ignore_index=True)
      predictions.append(pred_price)

  # 3. 绘图并保存到 OUTPUT_DIR
  plt.figure(figsize=(12, 6))

  # 提取展示用的历史区间
  hist_show = full_df.iloc[max(0, split_idx - 30): split_idx + 1]

  # 绘制历史线：黑色 (已知数据)
  plt.plot(hist_show['trade_date_dt'], hist_show['close'],
           label='Historical (Known)', color='black', marker='o', markersize=4, linewidth=2)

  # 绘制真实未来线：蓝色 (实际走势)
  plt.plot(real_future['trade_date_dt'], real_future['close'],
           label='Real Market (Actual)', color='blue', marker='o', markersize=4, linewidth=2)

  # 绘制模型预测线：红色 (预测走势)
  plt.plot(real_future['trade_date_dt'], predictions,
           label='Model Forecast (Predicted)', color='red', linestyle='--', marker='s', markersize=5)

  # 细节装饰
  plt.title(f"Backtest Analysis: {symbol} | Training Ends: {train_end_date}", fontsize=14)
  plt.xlabel("Date", fontsize=12)
  plt.ylabel("Price", fontsize=12)
  plt.legend(loc='best')
  plt.grid(True, linestyle=':', alpha=0.6)
  plt.xticks(rotation=45)

  # 在截断点增加一条垂直辅助线，区分“已知”和“预测”
  plt.axvline(pd.to_datetime(train_end_date), color='gray', linestyle='-.', alpha=0.5)
  plt.text(pd.to_datetime(train_end_date), plt.ylim()[0], ' Train End', color='gray', fontsize=10)

  plt.tight_layout()

  # 保存图片到 OUTPUT_DIR，使用 300 DPI 高清输出
  plot_filename = f"{symbol}_{model_suffix}_backtest.png"
  plot_path = os.path.join(cfg.OUTPUT_DIR, plot_filename)
  plt.savefig(plot_path, dpi=300)

  print(f"\n[SUCCESS] 回测对比图已保存至: {plot_path}")
  plt.show()


if __name__ == "__main__":
  run_backtest("600879", train_end_date="20260310", test_days=10)
