import os
from config import Config
from data_loader import DataManager
from model import StockLSTM
from trainer import Trainer


def main():
  # 1. 配置加载
  cfg = Config()
  symbol = "600879"  # 确保 data/train_data/600879.csv 存在

  # 2. 数据准备
  mgr = DataManager(cfg)
  try:
    train_loader = mgr.get_dataloader(symbol)
  except Exception as e:
    print(f"[ERROR] 数据加载失败: {e}")
    return

  # 3. 初始化模型
  # 注意：INPUT_DIM 会根据 config.FEATURE_COLS 自动计算
  model = StockLSTM(cfg)

  # 4. 训练
  trainer = Trainer(cfg, model)
  trainer.train_single_stock(symbol, train_loader, model_name_suffix="lstm")

  print("\n[FINISH] 训练任务已完成。你现在可以运行预测脚本了。")


if __name__ == "__main__":
  main()
