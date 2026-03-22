import os
from config import Config
from data_loader import DataManager
from model import get_model  # 只需要导入这个工厂函数
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
  model = get_model(cfg)

  # 4. 训练
  trainer = Trainer(cfg, model)
  # 这里的后缀也直接引用配置，保持一致性
  trainer.train_single_stock(symbol, train_loader, model_name_suffix=cfg.MODEL_NAME.lower())

  print("\n[FINISH] 训练任务已完成。你现在可以运行预测脚本了。")


if __name__ == "__main__":
  main()
