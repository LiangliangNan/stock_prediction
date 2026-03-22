"""
================================================================================
MODULE: Stock Backtest Engine (Sliding Window / Rolling Inference)
DESCRIPTION:
    本模块实现了一个工业级的“滑动窗口”闭环回测系统。
    解决了单次预测“不具统计意义”的问题。通过模拟真实的实盘调仓逻辑，
    让模型在一段长周期内，每隔固定的“持仓期（Hold Period）”重新基于最新真实数据进行预测。

CORE LOGIC & ARCHITECTURE:
    1. 训练数据隔离与模型复用 (Data Isolation & Model Reuse):
       - `train_end_date`: 物理截断点。
       - 支持 `always_train` 参数，默认复用已存在的本地权重，提升回测效率。

    2. 滑动窗口调仓 (Sliding Window Inference):
       - 引入 `test_start_date`, `test_end_date` 和 `hold_period`。
       - 模拟量化基金“周末跑模型定策略，周内持仓不动，下周末再跑模型”的核心逻辑。

    3. 局部快照落盘 (Segment Snapshot):
       - 每次完成 `hold_period` 天的推演后，自动截取包含局部历史上下文的独立预测图，
         并按严谨的时间戳命名规则落盘，便于后续归因分析。

    4. 性能指标对齐 (Performance Alignment):
       - 方向准确率 (Dir Accuracy): 基于每个决策周期的真实起跑点计算，剔除换仓跳跃干扰。
       - 模拟收益率 (Simulated Return): 模拟“预测上涨则持有”策略的真实累计收益。

AUTHOR: Gemini AI Assistant
DATE: 2026-03-22
================================================================================
"""

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib
from datetime import timedelta

# 内部模块导入
from config import Config
from data_loader import DataManager
from model import StockLSTM
from trainer import Trainer


class BacktestEngine:
  """
  股票回测引擎：支持滑动窗口多段预测的实战级版本
  """

  def __init__(self, symbol, train_end_date, test_start_date, test_end_date, hold_period=5):
    self.cfg = Config()
    self.symbol = symbol
    self.train_end_date = pd.to_datetime(train_end_date)
    self.test_start_date = pd.to_datetime(test_start_date)
    self.test_end_date = pd.to_datetime(test_end_date)
    self.hold_period = hold_period
    self.model_suffix = f"trainend_{train_end_date}_lstm"

    # 结果存储
    self.real_future = None
    self.predictions = []
    self.history_show = None
    self.last_known_price = 0
    self.last_known_date = None

    # 存储基础DataFrame供切片使用
    self.full_df = None

  def prepare_and_train(self, always_train=False):
    """
    [逻辑阶段 1] 数据隔离与模型复用/训练
    """
    model_filename = f"{self.symbol}_{self.model_suffix}.pth"
    model_path = os.path.join(self.cfg.MODEL_DIR, model_filename)

    self.model = StockLSTM(self.cfg)
    trainer = Trainer(self.cfg, self.model)

    if os.path.exists(model_path) and not always_train:
        print(f"\n[INFO] 发现已存在模型权重，跳过训练直接复用: {model_filename}")
        trainer.load_model(model_filename)
    else:
        print(f"\n[INFO] 开始训练模型 (always_train={always_train} 或本地无缓存)...")
        mgr = DataManager(self.cfg)
        train_loader = mgr.get_dataloader(self.symbol, train_end_date=self.train_end_date.strftime('%Y%m%d'))
        trainer.train_single_stock(self.symbol, train_loader, model_name_suffix=self.model_suffix)

    return self

  def _prepare_data_context(self):
    """
    [逻辑阶段 2] 全局上下文与坐标系建立
    """
    file_path = os.path.join(self.cfg.RAW_DATA_DIR, f"{self.symbol}.csv")
    df = pd.read_csv(file_path, header=0,
                     names=['ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'change', 'pct_chg', 'vol'])
    df['trade_date_dt'] = pd.to_datetime(df['trade_date'].astype(str))
    self.full_df = df.sort_values('trade_date_dt').reset_index(drop=True)

    # 定位回测起始点
    start_indices = self.full_df[self.full_df['trade_date_dt'] <= self.test_start_date].index
    if len(start_indices) == 0:
      raise ValueError(f"起始测试日期 {self.test_start_date} 超出数据范围")
    start_idx = start_indices[-1]

    # 定位回测结束点
    end_indices = self.full_df[self.full_df['trade_date_dt'] <= self.test_end_date].index
    end_idx = end_indices[-1] if len(end_indices) > 0 else len(self.full_df) - 1

    self.history_show = self.full_df.iloc[max(0, start_idx - 30): start_idx + 1]
    self.real_future = self.full_df.iloc[start_idx + 1: end_idx + 1].copy()

    self.last_known_date = self.full_df.iloc[start_idx]['trade_date_dt']
    self.last_known_price = self.full_df.iloc[start_idx]['close']

    return start_idx, end_idx

  def _plot_single_window(self, current_idx, chunk_dates, chunk_preds):
    """
    绘制并保存单次换仓期间的局部预测走势图
    """
    hist_df = self.full_df.iloc[max(0, current_idx - 15) : current_idx + 1]
    last_hist_date = hist_df['trade_date_dt'].iloc[-1]
    last_hist_price = hist_df['close'].iloc[-1]

    actual_df = self.full_df.iloc[current_idx + 1 : current_idx + 1 + len(chunk_preds)]

    p_dates = [last_hist_date] + chunk_dates
    p_preds = [last_hist_price] + chunk_preds

    plt.figure(figsize=(10, 5))
    plt.plot(hist_df['trade_date_dt'], hist_df['close'], color='black', marker='o', markersize=4, label='History Context')

    if not actual_df.empty:
        real_dates = [last_hist_date] + actual_df['trade_date_dt'].tolist()
        real_prices = [last_hist_price] + actual_df['close'].tolist()
        plt.plot(real_dates, real_prices, color='blue', marker='o', markersize=4, label='Actual Market', markevery=range(1, len(real_dates)))

    plt.plot(p_dates, p_preds, color='red', linestyle='--', marker='s', markersize=5, label=f'Forecast ({self.hold_period} Days)', markevery=range(1, len(p_dates)))

    label_text = f"{last_hist_date.strftime('%Y-%m-%d')}\nPrice: {last_hist_price:.2f}"
    plt.annotate(label_text, xy=(last_hist_date, last_hist_price),
                 xytext=(10, 20), textcoords='offset points',
                 arrowprops=dict(arrowstyle='->', color='green'))

    plt.title(f"Segment Forecast: {self.symbol} | Predict Start: {chunk_dates[0].strftime('%Y-%m-%d')}", fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    train_end_str = self.train_end_date.strftime('%Y%m%d')
    start_str = chunk_dates[0].strftime('%Y%m%d')
    fname = f"{self.symbol}_rolling_backtest_trainend_{train_end_str}_pred_start_{start_str}_hold_{self.hold_period}.png"

    out_path = os.path.join(self.cfg.OUTPUT_DIR, fname)
    plt.savefig(out_path, dpi=300)
    plt.close()

  def run_inference(self):
    """
    [逻辑阶段 3] 滑动窗口多段推理核心引擎
    """
    start_idx, end_idx = self._prepare_data_context()
    scaler = joblib.load(os.path.join(self.cfg.SCALER_DIR, f"{self.symbol}_scaler.pkl"))
    self.predictions = []

    self.model.eval()
    with torch.no_grad():
      current_idx = start_idx

      while current_idx < end_idx:
        chunk_end_idx = min(current_idx + self.hold_period, end_idx)
        days_to_predict = chunk_end_idx - current_idx

        current_df = self.full_df.iloc[current_idx - self.cfg.SEQ_LEN - 30: current_idx + 1].copy()
        chunk_preds = []
        chunk_dates = []

        for i in range(days_to_predict):
          # 1. 重算动态指标
          current_df['ma5'] = current_df['close'].rolling(5).mean()
          current_df['ma20'] = current_df['close'].rolling(20).mean()
          current_df['v_ma5'] = current_df['vol'].rolling(5).mean()
          temp_df = current_df.bfill().ffill()

          # 2. 特征提取与预测
          last_seq = temp_df[self.cfg.FEATURE_COLS].tail(self.cfg.SEQ_LEN).values
          scaled_input = scaler.transform(last_seq)
          input_tensor = torch.FloatTensor(scaled_input).unsqueeze(0).to(self.cfg.DEVICE)
          pred_norm = self.model(input_tensor)

          # 3. 逆缩放
          dummy = np.zeros((1, len(self.cfg.FEATURE_COLS)))
          target_idx = self.cfg.FEATURE_COLS.index(self.cfg.TARGET_COL)
          dummy[0, target_idx] = pred_norm.item()
          pred_price = scaler.inverse_transform(dummy)[0, target_idx]

          # 4. 日期对齐与闭环喂入
          next_trade_idx = current_idx + 1 + i
          if next_trade_idx < len(self.full_df):
              pred_date = self.full_df.iloc[next_trade_idx]['trade_date_dt']
          else:
              pred_date = current_df['trade_date_dt'].iloc[-1] + timedelta(days=1)

          new_row = {
            'trade_date_dt': pred_date,
            'close': pred_price, 'open': pred_price, 'high': pred_price, 'low': pred_price,
            'vol': current_df['vol'].mean()
          }
          current_df = pd.concat([current_df, pd.DataFrame([new_row])], ignore_index=True)

          self.predictions.append(pred_price)
          chunk_preds.append(pred_price)
          chunk_dates.append(pred_date)

        if len(chunk_preds) > 0:
            self._plot_single_window(current_idx, chunk_dates, chunk_preds)

        current_idx += self.hold_period

    return self

  def evaluate_and_plot(self):
    """
    [逻辑阶段 4] 性能指标对齐与全局可视化绘制
    """
    actuals = self.real_future['close'].values
    preds = np.array(self.predictions)
    total_len = len(preds)

    # --- [核心修改] 严谨的方向准确率与模拟收益率计算 ---
    correct_dirs = 0
    simulated_returns = 1.0  # 初始资金系数 1.0

    for i in range(0, total_len, self.hold_period):
        # 确定本段的起跑点价格 (真实价格)
        start_price = self.last_known_price if i == 0 else actuals[i-1]

        # 截取本段预测和真实走势
        chunk_preds = preds[i : i + self.hold_period]
        chunk_actuals = actuals[i : i + self.hold_period]

        # A. 计算方向一致性 (基于段内真实起点的增量对比)
        ref_preds = np.insert(chunk_preds, 0, start_price)
        ref_actuals = np.insert(chunk_actuals, 0, start_price)
        p_dir = (np.diff(ref_preds) >= 0).astype(int)
        a_dir = (np.diff(ref_actuals) >= 0).astype(int)
        correct_dirs += np.sum(p_dir == a_dir)

        # B. 计算模拟交易收益 (策略：如果模型预判 5 天后比当前涨，则买入持有)
        # 取本段最后一个预测值对比起点真实价格
        if chunk_preds[-1] > start_price:
            # 真实收益 = 本段末位真实价格 / 起点真实价格
            segment_real_return = chunk_actuals[-1] / start_price
            simulated_returns *= segment_real_return

    acc = (correct_dirs / total_len) * 100
    mae = np.mean(np.abs(actuals - preds))
    total_return_pct = (simulated_returns - 1) * 100

    # 辅助计算：基准收益 (Benchmark: 买入不动的真实涨跌)
    benchmark_return = (actuals[-1] / self.last_known_price - 1) * 100

    # --- 绘图 ---
    plt.figure(figsize=(12, 6))
    plt.plot(self.history_show['trade_date_dt'], self.history_show['close'],
             label='Historical', color='black', marker='o', markersize=4)

    p_dates = [self.last_known_date] + self.real_future['trade_date_dt'].tolist()
    p_real = [self.last_known_price] + actuals.tolist()
    p_pred = [self.last_known_price] + self.predictions

    plt.plot(p_dates, p_real, label='Actual Market', color='blue', marker='o', markevery=range(1, len(p_dates)))
    plt.plot(p_dates, p_pred, label=f'LSTM Forecast (Hold {self.hold_period} days)', color='red', linestyle='--',
             marker='s', markevery=range(1, len(p_dates)))

    label_text = f"{self.last_known_date.strftime('%Y-%m-%d')}\nPrice: {self.last_known_price:.2f}"
    plt.annotate(label_text, xy=(self.last_known_date, self.last_known_price),
                 xytext=(10, 20), textcoords='offset points',
                 arrowprops=dict(arrowstyle='->', color='green'))

    # 统计信息框 (集成新指标)
    stats_text = (f'Dir Accuracy: {acc:.1f}%\n'
                  f'MAE: {mae:.3f}\n'
                  f'Simulated Return: {total_return_pct:+.2f}%\n'
                  f'Benchmark: {benchmark_return:+.2f}%')

    plt.gca().text(0.02, 0.96, stats_text, transform=plt.gca().transAxes,
                   fontsize=10, verticalalignment='top', fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    plt.title(f"Rolling Backtest: {self.symbol} | Train End: {self.train_end_date.strftime('%Y-%m-%d')}")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend(loc='lower right')
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.xticks(rotation=45)
    plt.tight_layout()

    train_end_str = self.train_end_date.strftime('%Y%m%d')
    out_path = os.path.join(self.cfg.OUTPUT_DIR, f"{self.symbol}_rolling_backtest_trainend_{train_end_str}_global.png")
    plt.savefig(out_path, dpi=300)

    print(f"\n[REPORT] 滚动回测区间: {self.test_start_date.strftime('%Y-%m-%d')} 至 {self.test_end_date.strftime('%Y-%m-%d')}")
    print(f"[REPORT] 全局汇总图表保存至: {out_path}")
    print(f"[REPORT] 分段方向准确率: {acc:.2f}% | 模拟策略收益: {total_return_pct:+.2f}% | 基准收益: {benchmark_return:+.2f}%")
    plt.show()


if __name__ == "__main__":
  engine = BacktestEngine(
      symbol="600879",
      train_end_date="20260310",
      test_start_date="20260310",
      test_end_date="20260320",
      hold_period=2
  )
  engine.prepare_and_train(always_train=False).run_inference().evaluate_and_plot()
