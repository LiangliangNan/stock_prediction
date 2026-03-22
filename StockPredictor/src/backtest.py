"""
================================================================================
MODULE: Stock Backtest Engine (LSTM-Based)
DESCRIPTION:
    本模块实现了一个闭环的滚动回测系统。其核心目标是在“完全隔离未来数据”的前提下，
    模拟模型在实战环境中的“自回归推理”能力，并量化预测趋势的准确性。

CORE LOGIC & ARCHITECTURE:
    1. 数据隔离 (Data Isolation):
       - 输入参数 `train_end_date` 作为物理截断点。
       - DataManager 仅加载该日期前的数据进行训练，确保模型未见过测试期内的任何价格。

    2. 自回归滚动推理 (Autoregressive Rolling Inference):
       - 不同于简单的单步预测，本模块采用“预测引导预测”模式。
       - 逻辑：T日的预测价格 P(T) 会被作为“真实价格”喂回 DataFrame，配合
         重新计算的动态技术指标（如 MA5, MA20），来生成 T+1 日的预测。
       - 这模拟了实战中无法获取未来真实均线、只能依赖自身预判的严苛环境。

    3. 动态特征重构 (Dynamic Feature Reconstruction):
       - 在推理循环内部，每一轮都会根据最新预测的价格，重新计算受价格影响的所有特征。
       - 确保特征向量 (Feature Vector) 的实时性和逻辑一致性。

    4. 三色多维对齐评估 (Tri-Color Multi-Dimensional Evaluation):
       - 历史区 (Black): 验证模型对过去波动的捕捉。
       - 真实未来 (Blue): 实际发生的市场走势（Benchmark）。
       - 模型预言 (Red): 纯粹依赖历史规律推演出的轨迹。
       - 评估指标：
         * 方向一致率 (Directional Accuracy): 衡量“涨跌方向”预判的胜率。
         * 平均绝对误差 (MAE): 衡量“价格点数”偏离的幅度。

FILE STRUCTURE:
    - BacktestEngine (Class): 主引擎，封装训练、推理、评估流程。
    - run_inference: 实现自回归循环的核心算子。
    - evaluate_and_plot: 实现可视化与性能报表生成。

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
  股票回测引擎：实现时间截断、模型重训、滚动推理及性能量化。
  """

  def __init__(self, symbol, train_end_date, test_days=10):
    self.cfg = Config()
    self.symbol = symbol
    self.train_end_date = pd.to_datetime(train_end_date)
    self.test_days = test_days
    self.model_suffix = f"trainend_{train_end_date}"

    # 结果存储
    self.real_future = None
    self.predictions = []
    self.history_show = None
    self.last_known_price = 0
    self.last_known_date = None

  def prepare_and_train(self):
    """
    [逻辑阶段 1] 数据隔离与模型训练
    确保模型在回测起点之前完全“失忆”，不接触任何未来测试数据。
    """
    mgr = DataManager(self.cfg)
    train_loader = mgr.get_dataloader(self.symbol, train_end_date=self.train_end_date.strftime('%Y%m%d'))

    self.model = StockLSTM(self.cfg)
    trainer = Trainer(self.cfg, self.model)

    # 执行训练 (会自动保存权重和 Loss 图)
    trainer.train_single_stock(self.symbol, train_loader, model_name_suffix=self.model_suffix)
    return self

  def _load_data_context(self):
    """
    [逻辑阶段 2] 上下文环境构建
    读取原始 CSV，定位截断点索引，切分历史展示区和真实未来对比区。
    """
    file_path = os.path.join(self.cfg.RAW_DATA_DIR, f"{self.symbol}.csv")
    df = pd.read_csv(file_path, header=0,
                     names=['ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'change', 'pct_chg', 'vol'])
    df['trade_date_dt'] = pd.to_datetime(df['trade_date'].astype(str))
    df = df.sort_values('trade_date_dt').reset_index(drop=True)

    # 定位截断点
    split_indices = df[df['trade_date_dt'] <= self.train_end_date].index
    if len(split_indices) == 0:
      raise ValueError(f"日期 {self.train_end_date} 超出数据范围")

    split_idx = split_indices[-1]

    # 提取绘图用的历史片段 (前30天) 和 真实的未来片段
    self.history_show = df.iloc[max(0, split_idx - 30): split_idx + 1]
    self.real_future = df.iloc[split_idx + 1: split_idx + 1 + self.test_days].copy()

    # 记录连接点信息
    self.last_known_date = df.iloc[split_idx]['trade_date_dt']
    self.last_known_price = df.iloc[split_idx]['close']

    # 返回推理初始缓冲区
    return df.iloc[split_idx - self.cfg.SEQ_LEN - 30: split_idx + 1].copy()

  def run_inference(self):
    """
    [逻辑阶段 3] 自回归滚动预测 (Autoregressive Loop)
    核心逻辑：模型预测出的 T+1 价格会被作为真实价格喂回给模型，用于预测 T+2。
    """
    current_df = self._load_data_context()
    scaler = joblib.load(os.path.join(self.cfg.SCALER_DIR, f"{self.symbol}_scaler.pkl"))
    self.predictions = []

    self.model.eval()
    with torch.no_grad():
      for _ in range(len(self.real_future)):
        # 重新计算动态技术指标 (MA/Vol)
        current_df['ma5'] = current_df['close'].rolling(5).mean()
        current_df['ma20'] = current_df['close'].rolling(20).mean()
        current_df['v_ma5'] = current_df['vol'].rolling(5).mean()
        temp_df = current_df.bfill().ffill()

        # 提取最后 SEQ_LEN 长度的特征向量
        last_seq = temp_df[self.cfg.FEATURE_COLS].tail(self.cfg.SEQ_LEN).values
        scaled_input = scaler.transform(last_seq)
        input_tensor = torch.FloatTensor(scaled_input).unsqueeze(0).to(self.cfg.DEVICE)

        # 模型预测与逆缩放
        pred_norm = self.model(input_tensor)

        # 逆缩放对齐逻辑
        dummy = np.zeros((1, len(self.cfg.FEATURE_COLS)))
        target_idx = self.cfg.FEATURE_COLS.index(self.cfg.TARGET_COL)
        dummy[0, target_idx] = pred_norm.item()
        pred_price = scaler.inverse_transform(dummy)[0, target_idx]

        # 构造新行，将预测值“伪装”成历史，喂入下一轮
        new_row = {
          'trade_date_dt': current_df['trade_date_dt'].iloc[-1] + timedelta(days=1),
          'close': pred_price, 'open': pred_price, 'high': pred_price, 'low': pred_price,
          'vol': current_df['vol'].mean()  # 交易量取均值模拟
        }
        current_df = pd.concat([current_df, pd.DataFrame([new_row])], ignore_index=True)
        self.predictions.append(pred_price)
    return self

  def evaluate_and_plot(self):
    """
    [逻辑阶段 4] 性能指标对齐与可视化绘制
    """
    actuals = self.real_future['close'].values
    preds = np.array(self.predictions)

    # --- 性能计算 ---
    # 涨跌方向判断 (对比前一日)
    act_dirs = (np.diff(np.insert(actuals, 0, self.last_known_price)) >= 0).astype(int)
    pre_dirs = (np.diff(np.insert(preds, 0, self.last_known_price)) >= 0).astype(int)
    acc = (act_dirs == pre_dirs).mean() * 100
    mae = np.mean(np.abs(actuals - preds))

    # --- 绘图 ---
    plt.figure(figsize=(12, 6))

    # A. 黑色：历史已知区间
    plt.plot(self.history_show['trade_date_dt'], self.history_show['close'],
             label='Historical', color='black', marker='o', markersize=4)

    # 准备连接点序列
    p_dates = [self.last_known_date] + self.real_future['trade_date_dt'].tolist()
    p_real = [self.last_known_price] + actuals.tolist()
    p_pred = [self.last_known_price] + self.predictions

    # B. 蓝色：真实未来 (不画第一个点的Marker)
    plt.plot(p_dates, p_real, label='Actual Market', color='blue', marker='o',
             markevery=range(1, len(p_dates)))

    # C. 红色：模型预言 (不画第一个点的Marker)
    plt.plot(p_dates, p_pred, label='LSTM Forecast', color='red', linestyle='--',
             marker='s', markevery=range(1, len(p_dates)))

    # 标注
    label_text = f"{self.last_known_date.strftime('%Y-%m-%d')}\nPrice: {self.last_known_price:.2f}"
    plt.annotate(label_text, xy=(self.last_known_date, self.last_known_price),
                 xytext=(10, 20), textcoords='offset points',
                 arrowprops=dict(arrowstyle='->', color='green'))

    plt.title(f"Backtest: {self.symbol} | Acc: {acc:.1f}% | MAE: {mae:.3f}")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.5)

    # 保存并打印
    out_path = os.path.join(self.cfg.OUTPUT_DIR, f"{self.symbol}_{self.model_suffix}.png")
    plt.savefig(out_path, dpi=300)
    print(f"\n[REPORT] 方向准确率: {acc:.2f}% | 平均价格偏差: {mae:.4f}")
    print(f"[REPORT] 图表保存至: {out_path}")
    plt.show()


# --- 主程序入口 ---
if __name__ == "__main__":
  engine = BacktestEngine(symbol="600879", train_end_date="20260310", test_days=10)
  engine.prepare_and_train().run_inference().evaluate_and_plot()
