"""
data.py

Data acquisition helpers for the stock_prediction package.
"""
from __future__ import annotations

import argparse
import random
import datetime
import sys
from pathlib import Path
from typing import Iterable, Sequence

from Ashare import *

# 强制将当前脚本所在目录加入 path，确保能找到同级的 Ashare.py
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

try:
    from .init import (
        NoneDataFrame,
        daily_path,
        pd,
        stock_data_queue,
        stock_list_queue,
        threading,
        time
    )
except ImportError:  # pragma: no cover
    import sys

    current_dir = Path(__file__).resolve().parent
    root_dir = current_dir.parent.parent
    src_dir = root_dir / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    from stock_prediction.init import (
        NoneDataFrame,
        daily_path,
        pd,
        stock_data_queue,
        stock_list_queue,
        threading,
        time,
    )


def _rename_first_column(frame: pd.DataFrame, target_name: str) -> pd.DataFrame:
    """Rename the first column without referencing locale specific headers."""

    if frame.columns.empty:
        return frame
    first = frame.columns[0]
    if first != target_name:
        frame = frame.rename(columns={first: target_name})
    return frame


def get_stock_list(file_path):
    """
    通用股票池解析函数：
    1. 忽略以 '#' 开头的注释行
    2. 忽略空行、表头及分隔线
    3. 自动补全 6 位代码
    返回: list of symbols
    """
    symbols = []
    import os
    if not os.path.exists(file_path):
        return symbols

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            stripped_line = line.strip()
            # 过滤逻辑：注释行、空行、表头、分割线
            if not stripped_line or stripped_line.startswith("#"):
                continue
            if "|" not in stripped_line or "代码" in stripped_line or "---" in stripped_line:
                continue

            parts = stripped_line.split("|")
            if len(parts) >= 2:
                # 提取代码并补齐 6 位
                symbol = parts[0].strip().zfill(6)
                symbols.append(symbol)
    return symbols


def _iterable_from_code(ts_code: str | Sequence[str]) -> Iterable[str]:
    if isinstance(ts_code, str):
        if ts_code:
            return [ts_code]
        return []
    return ts_code


def get_stock_data(ts_code: Sequence[str] | str = "", save: bool = True, start_code: str = "",
                   save_path: Path | str = ""):
    """
    使用 Ashare 接口下载历史数据，并自动适配字段格式。
    支持增量更新：自动检测本地文件日期，仅抓取缺失的最新数据并合并。
    采用“局部计算+头部拼接”逻辑：仅计算新抓取数据的指标，然后覆盖拼接至旧数据，提高效率。
    """
    if isinstance(save_path, str):
        save_path = Path(save_path)

    # 1. 获取待处理的股票列表
    stock_list = list(_iterable_from_code(ts_code))
    if not stock_list:
        pool_file = "/Users/lnan/Documents/Projects/alpha/data/pool_core.txt"
        print(f"📂 正在从 {pool_file} 加载股票池...")
        stock_list = get_stock_list(pool_file)

    if not stock_list:
        print("💡 提示: 未发现有效股票代码，请检查路径")
        return

    # 断点续传逻辑
    if start_code:
        try:
            stock_list = stock_list[stock_list.index(start_code):]
        except ValueError:
            pass

    total_count = len(stock_list)
    today_str = datetime.datetime.now().strftime('%Y-%m-%d')
    for i, code in enumerate(stock_list, 1):  # i 从 1 开始计数
        try:
            # --- A. 初始化参数与增量检测 ---
            if not (code.startswith('sh') or code.startswith('sz')):
                api_code = ('sh' + code) if code.startswith('6') else ('sz' + code)
            else:
                api_code = code

            file_full_path = save_path / f"{code}.csv"
            existing_df = None
            fetch_count = 2000

            if save and file_full_path.exists():
                try:
                    existing_df = pd.read_csv(file_full_path)
                    if not existing_df.empty:
                        existing_df['trade_date'] = existing_df['trade_date'].astype(str)
                        last_date_str = str(existing_df['trade_date'].max())
                        last_date = datetime.datetime.strptime(last_date_str, '%Y%m%d')
                        days_diff = (datetime.datetime.now() - last_date).days
                        if days_diff <= 0:
                            progress_pct = (i / total_count) * 100
                            print(f"[{progress_pct:6.1f}%] {i:4d}/{total_count} ⏭️  [{code}] 已是最新，跳过")
                            continue
                        fetch_count = max(days_diff + 5, 5)
                except Exception:
                    existing_df = None

            # --- B. 调用接口并获取数据 ---
            df_new = None
            try:
                # 获取原始行情
                df_new = get_price(api_code, frequency='1d', count=fetch_count, end_date=today_str)
            except Exception as e:
                print(f"⚠️ {code} 接口调用失败: {e}")
                raise e

            if df_new is None or df_new.empty:
                continue

            # --- C. 数据清洗与指标计算 (修复核心重命名逻辑) ---
            df_new = df_new.copy()

            # Ashare 返回的日期在 Index 中，重命名 index 为 trade_date 并转为列
            df_new.index.name = 'trade_date'
            df_new.reset_index(inplace=True)

            # 统一列名映射：依赖原始列名而非索引位置
            # Ashare 标准输出列为: ['trade_date', 'open', 'high', 'low', 'close', 'volume']
            rename_map = {
                "volume": "vol",
                "Volume": "vol"  # 兼容部分版本可能出现的大写
            }
            df_new.rename(columns=rename_map, inplace=True)

            df_new.insert(0, "ts_code", code)
            df_new["trade_date"] = pd.to_datetime(df_new["trade_date"]).dt.strftime("%Y%m%d")

            # 计算局部指标
            df_new.sort_values(by="trade_date", ascending=True, inplace=True)
            if len(df_new) > 1:
                # 确保计算使用的是 float
                df_new["close"] = df_new["close"].astype(float)
                df_new["change"] = df_new["close"].diff()
                df_new["pct_change"] = df_new["close"].pct_change() * 100
                df_new.dropna(subset=['change'], inplace=True)
            else:
                df_new["change"] = 0.0
                df_new["pct_change"] = 0.0

            # 转为降序用于后续逻辑
            df_new.sort_values(by="trade_date", ascending=False, inplace=True)

            # --- D. 合并旧数据 ---
            if existing_df is not None:
                existing_df = existing_df.dropna(how='all', axis=1)
                combined_df = pd.concat([df_new, existing_df], axis=0, ignore_index=True)
                combined_df.drop_duplicates(subset=['trade_date'], keep='first', inplace=True)
                final_df = combined_df
            else:
                final_df = df_new

            # --- 精准拦截逻辑 ---
            if existing_df is not None and len(final_df) == len(existing_df):
                progress_pct = (i / total_count) * 100
                print(f"[{progress_pct:6.1f}%] {i:4d}/{total_count} ℹ️ [{code}] 数据日期相同，无需更新写入")
                continue

            # 统一输出列顺序
            final_df.sort_values(by="trade_date", ascending=False, inplace=True)
            final_df = final_df.reindex(columns=[
                "ts_code", "trade_date", "open", "high", "low", "close",
                "change", "pct_change", "vol"
            ])

            # --- E. 保存文件 ---
            if save:
                save_path.mkdir(parents=True, exist_ok=True)
                final_df.to_csv(file_full_path, index=False)

                progress_pct = (i / total_count) * 100
                latest_date = df_new['trade_date'].iloc[0]
                latest_close = df_new['close'].iloc[0]
                print(f"[{progress_pct:6.1f}%] {i:4d}/{total_count} ✅ [{code}] 抓取成功 | 日期: {latest_date} | 收盘: {latest_close:7.2f} | 数量: {len(df_new)}")
            else:
                stock_data_queue.put(final_df)
                if len(stock_list) == 1: return final_df

        except Exception as exc:
            print(f"[{ (i/total_count)*100:6.1f}%] ❌ 股票 {code} 失败: {str(exc)[:100]}")
            time.sleep(2)
            continue

        time.sleep(random.uniform(0.1, 0.3))

    return None


def main() -> None:
    """Command-line entry point for fetching quote data."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--code", default="", type=str, help="单个股票代码")
    # start_code：执行进度的“断点续传”。这个参数的作用是“从哪只股票开始运行”。
    # 应用场景：假设你有 4000 只股票要下载，运行到第 2000 只（比如代码是 600123）时，你的网络断了或者程序崩溃了。
    # 如果不使用它：你重新运行脚本，它会从第 1 只股票开始重新检查，虽然有增量逻辑不会重复下载，但遍历 2000 个文件的磁盘 IO 依然很慢。
    parser.add_argument("--start_code", default="", type=str, help="从哪个代码开始断点续传") # 程序中途崩溃，不想从第一只股票重新排队。

    args = parser.parse_args()
    get_stock_data(args.code, save=True, save_path=daily_path, start_code=args.start_code)


def sort_csv_files_ascending(folder_path):
  """
  遍历指定文件夹下的所有 CSV 文件，并按日期列进行物理升序排序。
  """
  data_path = Path(folder_path)
  if not data_path.exists():
    print(f"错误: 路径 {folder_path} 不存在")
    return

  processed_count = 0
  for file in data_path.glob("*.csv"):
    try:
      df = pd.read_csv(file)
      # 自动匹配日期列名（兼容 Tushare 的 trade_date 或其他来源的 Date）
      date_col = 'trade_date' if 'trade_date' in df.columns else ('Date' if 'Date' in df.columns else None)

      if date_col:
        # 核心：按日期从小到大（升序）排列
        df[date_col] = pd.to_datetime(df[date_col].astype(str))  # 确保是日期格式
        df = df.sort_values(date_col, ascending=True).reset_index(drop=True)

        # 写回原文件，index=False 避免多出一列索引
        df.to_csv(file, index=False, encoding='utf-8')
        processed_count += 1
        print(f"【成功】已物理翻转并对齐升序: {file.name}")
      else:
        print(f"【跳过】未在 {file.name} 中找到日期列")
    except Exception as e:
      print(f"【失败】处理 {file.name} 时出错: {e}")

  print(f"\n处理完成！共处理 {processed_count} 个文件。")


if __name__ == "__main__":
    main()
    # sort_csv_files_ascending("../../stock_daily")
