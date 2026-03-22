StockPredictor Pro (Clean-Stream版) 完整架构：
1. 核心目录结构。遵循“数据与逻辑分离”原则，确保修改模型时不会弄乱数据处理逻辑。 

StockProject
├── data/
│   ├── train_data/        # 存放所有股票的升序 CSV (如 600879.csv)
│   └── scaler_params/     # 存放归一化参数 (每个股票对应的 mean/std)
├── models/
│   └── checkpoints/       # 存放训练好的 .pth 模型文件
├── src/
│   ├── config.py          # 唯一配置中心：SEQ_LEN, 特征列表, 隐藏层维度
│   ├── data_loader.py     # 模块A：读取CSV, 构造 (X, y) Tensor
│   ├── model.py           # 模块B：神经网络定义 (LSTM/Transformer)
│   ├── trainer.py         # 模块C：训练逻辑 (支持 Early Stopping)
│   └── predictor.py       # 模块D：滚动预测明天、后天、大后天...
└── run.py                 # 总控入口：--mode train 或 --mode predict
2. 模块化详细设计
- 模块 A：data_loader.py (数据工厂)。功能：加载 CSV，计算技术指标（如 MA5, RSI），并执行 fit_transform 归一化。
  关键点：它必须保存每个股票的 StandardScaler 对象，因为预测时反归一化必须用训练时的参数。
- 模块 B：model.py (模型中心)。使用带 Dropout 的多层 LSTM 或 GRU。
  输入输出：输入维度 [Batch, 60, Features]，输出维度 [Batch, 1] (预测明天的收盘价)。
- 模块 C：predictor.py (滚动推理引擎)。采用“贪婪填充”逻辑：
  * Seed: 取历史最后的 60 天数据。
  * Predict: 模型输出 $T+1$ 的价格。
  * Update: 将 $T+1$ 插入窗口末尾，
  * 移除窗口最前面的 $T-59$ 天。
  * Recalculate: 如果特征包含 MA5，在加入预测值后，需动态重新计算 MA5，然后再进行 $T+2$ 的预测。
3. 核心流程图 (Workflow)
4. 架构的技术亮点
- 特征自动对齐：通过 config.py 定义 FEATURE_COLS，确保训练和预测时特征列的顺序索引完全一致，避免“张冠李戴”。
- 持久化归一化：预测不再使用全量数据重新计算均值，而是加载训练时的 scaler。这能解决“预测值受历史窗口长度影响”的 Bug。
- 解耦评估：专门的 metrics.py（可选）计算 MAE、RMSE 以及最重要的方向准确率 (Directional Accuracy)。
