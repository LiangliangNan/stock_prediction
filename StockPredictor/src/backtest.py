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

DATE: 2026-03-22
================================================================================
"""

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates # 日期处理模块
import os
import joblib
from datetime import timedelta
from data_loader import DataManager

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
        print(f"\n[INFO] 开始训练模型 (always_train={always_train} 或本地无缓存)...", flush=True)
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

  def _setup_fine_grid(self, ax):
    """
    配置加密网格：主刻度5天显示文字，次刻度每天显示虚线
    """
    # 设置主要刻度：每5个交易日显示一次日期文字
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

    # 设置次要刻度：每一天一个定位点 (加密核心)
    ax.xaxis.set_minor_locator(mdates.DayLocator(interval=1))

    # 绘制网格：主网格实线，次网格更细更淡的虚线
    ax.grid(which='major', linestyle='-', linewidth='0.5', color='gray', alpha=0.5)
    ax.grid(which='minor', linestyle=':', linewidth='0.3', color='silver', alpha=0.8)

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
    ax = plt.gca()  # <-- 获取当前坐标轴对象
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

    self._setup_fine_grid(ax)
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
    执行回测推理逻辑 (优化版：按 hold_period 跳跃推理，消除计算浪费)
    设计思路：
        1. 仅在决策日（i, i+hold_period...）提取特征并推理。
        2. 如果模型是单步预测，则在决策日基于当时信息预测未来价格。
    """
    self._prepare_data_context()

    from data_loader import DataManager
    dm = DataManager(self.cfg)

    # 1. 加载并清洗基础数据
    file_path = os.path.join(self.cfg.RAW_DATA_DIR, f"{self.symbol}.csv")
    df = pd.read_csv(file_path, header=0, names=[
      'ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'change', 'pct_chg', 'vol'
    ])
    df['trade_date_dt'] = pd.to_datetime(df['trade_date'].astype(str))
    df = df.sort_values('trade_date_dt').reset_index(drop=True)

    # 2. 调用特征工厂
    df = dm._extract_features(df)
    self.full_df = df

    # 3. 定位回测区间数据
    test_df = df[df['trade_date_dt'] >= pd.to_datetime(self.test_start_date)].copy()
    if test_df.empty:
      raise ValueError(f"回测起始日期 {self.test_start_date} 超出数据范围")

    # 4. 加载归一化器
    scaler_path = os.path.join(self.cfg.SCALER_DIR, f"{self.symbol}_scaler.pkl")
    scaler = joblib.load(scaler_path)

    print(f"[INFO] 开始按持仓期({self.hold_period}天)进行跳跃回测: {self.symbol}...")
    self.predictions = []

    self.model.eval()
    with torch.no_grad():
      # --- [核心修改]：i 以 self.hold_period 为步长跳跃 ---
      for i in range(0, len(test_df), self.hold_period):
        current_date = test_df.iloc[i]['trade_date_dt']
        idx_in_full = df[df['trade_date_dt'] == current_date].index[0]

        # 提取决策日当天的特征序列
        if idx_in_full < self.cfg.SEQ_LEN:
          pred_price = test_df.iloc[i]['close']
        else:
          seq_df = df.iloc[idx_in_full - self.cfg.SEQ_LEN: idx_in_full]
          last_seq = seq_df[self.cfg.FEATURE_COLS].values

          scaled_seq = scaler.transform(last_seq)
          input_tensor = torch.FloatTensor(scaled_seq).unsqueeze(0).to(self.cfg.DEVICE)
          pred_norm = self.model(input_tensor).item()

          target_idx = self.cfg.FEATURE_COLS.index(self.cfg.TARGET_COL)
          dummy = np.zeros((1, len(self.cfg.FEATURE_COLS)))
          dummy[0, target_idx] = pred_norm
          pred_price = scaler.inverse_transform(dummy)[0, target_idx]

        # --- [逻辑对齐]：模拟决策 ---
        # 决策日预测了一个价格 pred_price，在整个 hold_period 期间，
        # 我们的“预期目标”保持不变，直到下一个决策日重新计算。
        for step in range(self.hold_period):
          if i + step < len(test_df):
            self.predictions.append(pred_price)

    print(f"[SUCCESS] 跳跃推理完成，有效决策点数: {int(np.ceil(len(test_df) / self.hold_period))}")
    return self

  def evaluate_and_plot(self):
    """
    [逻辑阶段 4] 性能指标对齐与全局可视化绘制
    """
    # 1. 长度交集对齐
    total_len = min(len(self.predictions), len(self.real_future))
    if total_len == 0:
      print("[ERROR] 没有足够的匹配数据，无法执行评估。")
      return self

    actuals = self.real_future['close'].values[:total_len]
    preds = np.array(self.predictions)[:total_len]
    aligned_dates = self.real_future['trade_date_dt'].tolist()[:total_len]

    # --- [核心修改 1]：显式创建全局画布，防止被局部绘图冲掉 ---
    fig_global = plt.figure(figsize=(12, 6))
    ax_global = plt.gca()

    # 2. 滚动计算指标并触发局部快照
    correct_dirs = 0
    simulated_returns = 1.0

    for i in range(0, total_len, self.hold_period):
      start_price = self.last_known_price if i == 0 else actuals[i - 1]
      chunk_preds = preds[i: i + self.hold_period]
      chunk_actuals = actuals[i: i + self.hold_period]
      if len(chunk_actuals) == 0: continue

      # --- [执行快照保存] ---
      current_decision_date = aligned_dates[i]
      temp_indices = self.full_df[self.full_df['trade_date_dt'] < current_decision_date].index
      if len(temp_indices) > 0:
        decision_idx_in_full = temp_indices[-1]
        chunk_dates = aligned_dates[i: i + self.hold_period]
        # 内部会执行 plt.close()，由于我们上面用了 fig_global 变量，这里不会影响全局句柄
        self._plot_single_window(decision_idx_in_full, chunk_dates, chunk_preds.tolist())

      # 重新激活全局画布（关键：确保后续 plot 回到 global 图上）
      plt.figure(fig_global.number)

      # A. 方向准确率
      ref_preds = np.insert(chunk_preds, 0, start_price)
      ref_actuals = np.insert(chunk_actuals, 0, start_price)
      if len(ref_preds) > 1:
        p_dir = (np.diff(ref_preds) >= 0).astype(int)
        a_dir = (np.diff(ref_actuals) >= 0).astype(int)
        correct_dirs += np.sum(p_dir == a_dir)

      # B. 模拟收益
      if chunk_preds[-1] > start_price:
        simulated_returns *= (chunk_actuals[-1] / start_price)

    # 4. 指标汇总计算
    acc = (correct_dirs / total_len) * 100
    mae = np.mean(np.abs(actuals - preds))
    total_return_pct = (simulated_returns - 1) * 100
    benchmark_return = (actuals[-1] / self.last_known_price - 1) * 100

    # --- [核心修改 2]：使用 ax_global 确保内容画在正确的画布上 ---
    ax_global.plot(self.history_show['trade_date_dt'], self.history_show['close'],
                   label='Historical', color='black', marker='o', markersize=4)

    p_dates = [self.last_known_date] + aligned_dates
    p_real = [self.last_known_price] + actuals.tolist()
    p_pred = [self.last_known_price] + preds.tolist()

    ax_global.plot(p_dates, p_real, label='Actual Market', color='blue', marker='o', markevery=range(1, len(p_dates)))
    ax_global.plot(p_dates, p_pred, label=f'LSTM Forecast (Hold {self.hold_period} days)', color='red', linestyle='--',
                   marker='s', markevery=range(1, len(p_dates)))

    label_text = f"{self.last_known_date.strftime('%Y-%m-%d')}\nPrice: {self.last_known_price:.2f}"
    ax_global.annotate(label_text, xy=(self.last_known_date, self.last_known_price),
                       xytext=(10, 20), textcoords='offset points',
                       arrowprops=dict(arrowstyle='->', color='green'))

    stats_text = (f'Dir Accuracy: {acc:.1f}%\n'
                  f'MAE: {mae:.3f}\n'
                  f'Simulated Return: {total_return_pct:+.2f}%\n'
                  f'Benchmark: {benchmark_return:+.2f}%')

    ax_global.text(0.02, 0.96, stats_text, transform=ax_global.transAxes,
                   fontsize=10, verticalalignment='top', fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    self._setup_fine_grid(ax_global)
    ax_global.set_title(f"Rolling Backtest: {self.symbol} | Aligned Samples: {total_len}")
    ax_global.legend(loc='lower left')
    plt.xticks(rotation=45)
    plt.tight_layout()

    # 保存并展示
    train_end_str = self.train_end_date.strftime('%Y%m%d')
    out_path = os.path.join(self.cfg.OUTPUT_DIR, f"{self.symbol}_rolling_backtest_trainend_{train_end_str}_global.png")
    plt.savefig(out_path, dpi=300)

    # 5. 最后展示
    plt.show()

    self._print_final_report(acc, mae, total_return_pct, benchmark_return, total_len)
    return self

  def _print_final_report(self, acc, mae, ret, bench, count):
    """ 抽取打印逻辑，使代码更清爽 """
    print(f"\n" + "=" * 50)
    print(f"[REPORT] 滚动回测完成 | 总样本量: {count}")
    print(f"[REPORT] 分段方向准确率: {acc:.2f}%")
    print(f"[REPORT] 模拟策略收益: {ret:+.2f}%")
    print(f"[REPORT] 基准收益: {bench:+.2f}%")
    print(f"[REPORT] MAE 误差: {mae:.4f}")
    print("=" * 50 + "\n")

if __name__ == "__main__":
  engine = BacktestEngine(
      symbol="600879",
      train_end_date="20260310",
      test_start_date="20260310",
      test_end_date="20260320",
      hold_period=2
  )
  engine.prepare_and_train(always_train=False).run_inference().evaluate_and_plot()
