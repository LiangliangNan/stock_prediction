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

  # 定位截断点
  split_indices = full_df[full_df['trade_date_dt'] <= pd.to_datetime(train_end_date)].index
  if len(split_indices) == 0:
    print(f"[ERROR] 找不到日期 {train_end_date}")
    return
  split_idx = split_indices[-1]

  # 记录截断点的数据点（用于连接线条）
  last_known_date = full_df.iloc[split_idx]['trade_date_dt']
  last_known_price = full_df.iloc[split_idx]['close']

  # 提取真实发生的未来数据
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

  # A. 绘制历史线：黑色 (包含最后一个点)
  plt.plot(hist_show['trade_date_dt'], hist_show['close'],
           label='Historical (Known)', color='black', marker='o', markersize=4, linewidth=2)

  # 准备连接点
  last_known_date = full_df.iloc[split_idx]['trade_date_dt']
  last_known_price = full_df.iloc[split_idx]['close']

  # --- 核心修正：确保日期和价格列表长度严格一致 ---

  # 蓝色真实线数据：[连接点] + 后面10天的真实日期和价格
  plot_real_dates = [last_known_date] + real_future['trade_date_dt'].tolist()
  plot_real_prices = [last_known_price] + real_future['close'].tolist()

  # 红色预测线数据：[连接点] + 后面10天的预测价格
  # 注意：日期必须使用和蓝色线完全一样的 plot_real_dates
  plot_pred_prices = [last_known_price] + predictions

  # B. 绘制真实未来线：蓝色 (使用 markevery 跳过第0个点，确保交汇处是黑色)
  plt.plot(plot_real_dates, plot_real_prices,
           label='Real Market (Actual)', color='blue', marker='o', markersize=4, linewidth=2,
           markevery=range(1, len(plot_real_dates)))

  # C. 绘制模型预测线：红色虚线 (同样跳过第0个点)
  plt.plot(plot_real_dates, plot_pred_prices,
           label='Model Forecast (Predicted)', color='red', linestyle='--', marker='s', markersize=5,
           markevery=range(1, len(plot_real_dates)))

  # D. 增加标注：显示截断点的价格和日期
  label_text = f"{last_known_date.strftime('%Y-%m-%d')}\nPrice: {last_known_price:.2f}"
  plt.annotate(label_text,
               xy=(last_known_date, last_known_price),
               xytext=(10, 15),  # 稍微调高偏移量以适应两行文字
               textcoords='offset points',
               arrowprops=dict(arrowstyle='->', color='black'))

  # 细节装饰
  plt.title(f"Backtest Analysis: {symbol} | Training Ends: {train_end_date}", fontsize=14)
  plt.xlabel("Date", fontsize=12)
  plt.ylabel("Price", fontsize=12)
  plt.legend(loc='best')
  plt.grid(True, linestyle=':', alpha=0.6)
  plt.xticks(rotation=45)

  # 垂直辅助线
  plt.axvline(last_known_date, color='gray', linestyle='-.', alpha=0.5)
  plt.text(last_known_date, plt.ylim()[0], ' Train End', color='gray', fontsize=10)

  plt.tight_layout()

  # 保存图片
  plot_filename = f"{symbol}_{model_suffix}_backtest.png"
  plot_path = os.path.join(cfg.OUTPUT_DIR, plot_filename)
  plt.savefig(plot_path, dpi=300)

  print(f"\n[SUCCESS] 回测对比图已保存至: {plot_path}")
  plt.show()


if __name__ == "__main__":
  run_backtest("600879", train_end_date="20260310", test_days=10)
