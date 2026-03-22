"""
================================================================================
MODULE: Stock Trend Predictor (Recursive Multi-step Inference)
DESCRIPTION:
    本模块负责利用训练好的模型进行未来 N 天的递归滚动预测。

DESIGN PHILOSOPHY (设计思路):
    1. 特征同步 (Feature Synchronization): 
       直接调用 DataManager 的 `_extract_features` 管道。这确保了预测阶段使用的特征
       与训练阶段完全一致，彻底解决了 "KeyError: features not in index" 问题。
    2. 递归闭环 (Recursive Feedback Loop): 
       每预测出未来一天的价格，将其作为“真实”价格喂回 DataFrame，重新触发特征管道
       计算衍生指标（如 MA、Bias、Returns），再进行下一天的预测。
    3. 零维护成本 (Zero Maintenance): 
       由于复用了 DataManager 的特征工厂，用户在测试新特征时只需修改 data_loader.py，
       本文件无需任何改动。

USAGE GUIDE (后续测试新特征的方法):
    1. 在 data_loader.py 的 [FEATURE FACTORY] 中添加新指标。
    2. 在 config.py 的 FEATURE_COLS 中注册新指标名。
    3. 直接运行 visualizer.py。本模块会自动识别新特征并参与计算。

DATE: 2026-03-22
================================================================================
"""

import torch
import pandas as pd
import numpy as np
import joblib
import os
from datetime import timedelta
from data_loader import DataManager  # 引入数据管理器以复用特征工厂


class Predictor:
  def __init__(self, config, model):
    self.config = config
    self.model = model.to(config.DEVICE)
    self.model.eval()
    # 实例化特征管理器
    self.dm = DataManager(self.config)

  def predict_future(self, symbol, days=5):
    scaler_path = os.path.join(self.config.SCALER_DIR, f"{symbol}_scaler.pkl")
    if not os.path.exists(scaler_path):
      raise FileNotFoundError(f"找不到 {symbol} 的归一化参数。")
    scaler = joblib.load(scaler_path)

    file_path = os.path.join(self.config.RAW_DATA_DIR, f"{symbol}.csv")
    # 读取原始 9 列数据
    df = pd.read_csv(file_path, header=0, names=[
      'ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'change', 'pct_chg', 'vol'
    ])
    df['trade_date'] = pd.to_datetime(df['trade_date'].astype(str))

    # 截取足够的长度以满足 SEQ_LEN 和 技术指标计算(如MA20)的初始需求
    current_df = df.tail(self.config.SEQ_LEN + 30).copy()
    predictions = []

    with torch.no_grad():
      for i in range(days):
        # --- [关键改动] 闭环调用 DataManager 的特征管道 ---
        # 这一步会自动计算 bias_5, v_ratio, returns, amplitude 等所有新特征
        temp_df = self.dm._extract_features(current_df)

        # 提取特征并归一化
        last_features = temp_df[self.config.FEATURE_COLS].tail(self.config.SEQ_LEN).values
        scaled_input = scaler.transform(last_features)
        input_tensor = torch.FloatTensor(scaled_input).unsqueeze(0).to(self.config.DEVICE)

        # 模型推理
        pred_normalized = self.model(input_tensor)

        # 反归一化
        dummy_row = np.zeros((1, len(self.config.FEATURE_COLS)))
        target_idx = self.config.FEATURE_COLS.index(self.config.TARGET_COL)
        dummy_row[0, target_idx] = pred_normalized.item()
        real_pred_price = scaler.inverse_transform(dummy_row)[0, target_idx]

        # 构造下一天数据
        last_date = current_df['trade_date'].iloc[-1]
        next_date = last_date + timedelta(days=1)

        new_row = {
          'ts_code': symbol, 'trade_date': next_date,
          'close': real_pred_price, 'open': real_pred_price,
          'high': real_pred_price, 'low': real_pred_price,
          'vol': current_df['vol'].mean()  # 递归预测时成交量取均值作为占位
        }

        # 更新 DataFrame 供下一次循环计算特征
        current_df = pd.concat([current_df, pd.DataFrame([new_row])], ignore_index=True)

        predictions.append({
          'date': next_date.strftime('%Y-%m-%d'),
          'predicted_close': round(real_pred_price, 2)
        })

    return pd.DataFrame(predictions)
