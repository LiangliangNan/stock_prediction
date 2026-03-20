"""Data acquisition helpers for the stock_prediction package."""
from __future__ import annotations

import argparse
import datetime
import random
import sys
from pathlib import Path
from typing import Iterable, Sequence

from  Ashare import *
# 强制将当前脚本所在目录加入 path，确保能找到同级的 Ashare.py
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

try:
    from .init import (
        TQDM_NCOLS,
        NoneDataFrame,
        daily_path,
        pd,
        stock_data_queue,
        stock_list_queue,
        threading,
        time,
        tqdm,
    )
except ImportError:  # pragma: no cover
    import sys

    current_dir = Path(__file__).resolve().parent
    root_dir = current_dir.parent.parent
    src_dir = root_dir / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    from stock_prediction.init import (
        TQDM_NCOLS,
        NoneDataFrame,
        daily_path,
        pd,
        stock_data_queue,
        stock_list_queue,
        threading,
        time,
        tqdm,
    )

try:
    import tushare as ts
except ImportError:
    ts = None

try:
    import akshare as ak
except ImportError:
    ak = None

try:
    import yfinance as yf
except ImportError:
    yf = None


class DataConfig:
    """Runtime switches controlling which upstream API is used."""

    def __init__(self) -> None:
        self.api = "akshare"
        self.adjust = "hfq"
        self.code = ""


config = DataConfig()


def set_adjust(adjust: str) -> None:
    """Update the adjustment flag used for downstream fetch operations."""

    config.adjust = adjust


def _rename_first_column(frame: pd.DataFrame, target_name: str) -> pd.DataFrame:
    """Rename the first column without referencing locale specific headers."""

    if frame.columns.empty:
        return frame
    first = frame.columns[0]
    if first != target_name:
        frame = frame.rename(columns={first: target_name})
    return frame


# def get_stock_list() -> Sequence[str]:
#     """Return a list of stock codes based on the configured API provider."""
#
#     if config.api == "tushare":
#         if ts is None:
#             raise ImportError("tushare not installed")
#         df = ts.pro_api().stock_basic(fields=["ts_code"])
#         stock_list = df["ts_code"].tolist()
#         stock_list_queue.put(stock_list)
#         return stock_list
#
#     if config.api == "akshare":
#         if ak is None:
#             raise ImportError("akshare not installed")
#         stock_frame = ak.stock_zh_a_spot_em()
#         stock_frame = _rename_first_column(stock_frame, "code")
#         stock_list = stock_frame["code"].astype(str).tolist()
#         stock_list_queue.put(stock_list)
#         return stock_list
#
#     raise ValueError(f"Unsupported api provider: {config.api}")

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


# def get_stock_data(ts_code: Sequence[str] | str = "", save: bool = True, start_code: str = "", save_path: Path | str = "", datediff: int = -1):
#     """Download historical bar data for the provided symbols."""
#
#     if isinstance(save_path, str):
#         save_path = Path(save_path)
#
#     if config.api == "tushare":
#         if ts is None:
#             raise ImportError("tushare not installed")
#
#         pro = ts.pro_api()
#         stock_list = list(_iterable_from_code(ts_code)) or get_stock_list()
#         if start_code:
#             stock_list = stock_list[stock_list.index(start_code):]
#         pbar = tqdm(total=len(stock_list), leave=False, ncols=TQDM_NCOLS) if save else None
#         lock = threading.Lock()
#
#         with lock:
#             adjust_suffix = f"_{config.adjust}" if config.adjust else ""
#             for code in stock_list:
#                 try:
#                     if config.adjust:
#                         fields = [
#                             "ts_code",
#                             "trade_date",
#                             f"open{adjust_suffix}",
#                             f"high{adjust_suffix}",
#                             f"low{adjust_suffix}",
#                             f"close{adjust_suffix}",
#                             f"pre_close{adjust_suffix}",
#                             "change",
#                             "pct_change",
#                             "vol",
#                             "amount",
#                         ]
#                         df = pro.stk_factor(ts_code=code, fields=fields)
#                         df.columns = [
#                             "ts_code",
#                             "trade_date",
#                             "open",
#                             "high",
#                             "low",
#                             "close",
#                             "pre_close",
#                             "change",
#                             "pct_change",
#                             "vol",
#                             "amount",
#                         ]
#                     else:
#                         df = pro.daily(ts_code=code, fields=[
#                             "ts_code",
#                             "trade_date",
#                             "open",
#                             "high",
#                             "low",
#                             "close",
#                             "pre_close",
#                             "change",
#                             "pct_chg",
#                             "vol",
#                             "amount",
#                         ])
#                         df = df.rename(columns={"pct_chg": "pct_change"})
#                     df = df.reindex(columns=[
#                         "ts_code",
#                         "trade_date",
#                         "open",
#                         "high",
#                         "low",
#                         "close",
#                         "change",
#                         "pct_change",
#                         "vol",
#                         "amount",
#                         "pre_close",
#                     ])
#                 except Exception as exc:  # pragma: no cover
#                     message = f"{code} {exc}"
#                     if save and pbar is not None:
#                         tqdm.write(message)
#                         pbar.update(1)
#                     else:
#                         print(message)
#                     continue
#
#                 time.sleep(random.uniform(31, 36))
#                 if save:
#                     save_path.mkdir(parents=True, exist_ok=True)
#                     df.to_csv(save_path / f"{code}.csv", index=False)
#                     if pbar is not None:
#                         pbar.update(1)
#                 else:
#                     stock_data_queue.put(df if not df.empty else NoneDataFrame)
#                     return df if not df.empty else None
#
#         if pbar is not None:
#             pbar.close()
#         return None
#
#     if config.api == "akshare":
#         if ak is None:
#             raise ImportError("akshare not installed")
#
#         stock_list = list(_iterable_from_code(ts_code)) or get_stock_list()
#         if start_code:
#             stock_list = stock_list[stock_list.index(start_code):]
#         pbar = tqdm(total=len(stock_list), leave=False, ncols=TQDM_NCOLS) if save else None
#         lock = threading.Lock()
#
#         with lock:
#             end_date = (datetime.datetime.now() + datetime.timedelta(days=datediff)).strftime("%Y%m%d")
#             for code in stock_list:
#                 try:
#                     df = ak.stock_zh_a_hist(symbol=code, period="daily", end_date=end_date, adjust=config.adjust)
#                     df.columns = [
#                         "trade_date",
#                         "ts_code",
#                         "open",
#                         "close",
#                         "high",
#                         "low",
#                         "vol",
#                         "amount",
#                         "amplitude",
#                         "pct_change",
#                         "change",
#                         "exchange_rate",
#                     ]
#                     columns = list(df.columns)
#                     columns[0], columns[1] = columns[1], columns[0]
#                     df = df[columns]
#                     df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.strftime("%Y%m%d")
#                     df.sort_values(by=["trade_date"], ascending=False, inplace=True)
#                     df = df.reindex(columns=[
#                         "ts_code",
#                         "trade_date",
#                         "open",
#                         "high",
#                         "low",
#                         "close",
#                         "change",
#                         "pct_change",
#                         "vol",
#                         "amount",
#                         "amplitude",
#                         "exchange_rate",
#                     ])
#                 except Exception as exc:  # pragma: no cover
#                     message = f"{code} {exc}"
#                     if save and pbar is not None:
#                         tqdm.write(message)
#                         pbar.update(1)
#                     else:
#                         print(message)
#                     if getattr(exc, "args", []) and isinstance(exc.args[0], Exception):
#                         inner = exc.args[0]
#                         text = str(inner)
#                         if "Connection aborted" in text or "Remote end closed connection" in text:
#                             break
#                     continue
#
#                 time.sleep(random.uniform(0.1, 0.9))
#                 if save:
#                     save_path.mkdir(parents=True, exist_ok=True)
#                     df.to_csv(save_path / f"{code}.csv", index=False)
#                     if pbar is not None:
#                         pbar.update(1)
#                 else:
#                     stock_data_queue.put(df if not df.empty else NoneDataFrame)
#                     return df if not df.empty else None
#
#         if pbar is not None:
#             pbar.close()
#         return None
#
#     if config.api == "yfinance":
#         if yf is None:
#             raise ImportError("yfinance not installed")
#
#         auto_adjust = back_adjust = False
#         if config.adjust == "qfq":
#             auto_adjust = True
#         elif config.adjust == "hfq":
#             auto_adjust = True
#             back_adjust = True
#
#         stock_list = list(_iterable_from_code(ts_code))
#         pbar = tqdm(total=len(stock_list), leave=False, ncols=TQDM_NCOLS) if save else None
#         lock = threading.Lock()
#
#         with lock:
#             for code in stock_list:
#                 try:
#                     df = yf.download(code, auto_adjust=auto_adjust, back_adjust=back_adjust)
#                     df.reset_index(inplace=True)
#                     df.insert(0, "ts_code", code)
#                     df.columns = [
#                         "ts_code",
#                         "trade_date",
#                         "open",
#                         "high",
#                         "low",
#                         "close",
#                         "vol",
#                     ]
#                     df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.strftime("%Y%m%d")
#                     df.sort_values(by=["trade_date"], ascending=False, inplace=True)
#                     df = df.reindex(columns=[
#                         "ts_code",
#                         "trade_date",
#                         "open",
#                         "high",
#                         "low",
#                         "close",
#                         "vol",
#                     ])
#                 except Exception as exc:  # pragma: no cover
#                     message = f"{code} {exc}"
#                     if save and pbar is not None:
#                         tqdm.write(message)
#                         pbar.update(1)
#                     else:
#                         print(message)
#                     continue
#
#                 if save:
#                     save_path.mkdir(parents=True, exist_ok=True)
#                     df.to_csv(save_path / f"{code}.csv", index=False)
#                     if pbar is not None:
#                         pbar.update(1)
#                 else:
#                     stock_data_queue.put(df if not df.empty else NoneDataFrame)
#                     return df if not df.empty else None
#
#         if pbar is not None:
#             pbar.close()
#         return None
#
#     raise ValueError(f"Unsupported api provider: {config.api}")


def get_stock_data(ts_code: Sequence[str] | str = "", save: bool = True, start_code: str = "",
                   save_path: Path | str = "", datediff: int = -1):
  """
  使用 Ashare 接口下载历史数据，并自动适配字段格式。
  解决了 Ashare 内部 6 vs 7 列报错及新浪接口被封禁导致的 JSONDecodeError。
  """
  import requests  # 确保导入请求库

  if isinstance(save_path, str):
    save_path = Path(save_path)

  # 1. 获取待处理的股票列表
  stock_list = list(_iterable_from_code(ts_code))
  if not stock_list:
    pool_file = "/Users/lnan/Documents/Projects/alpha/data/pool_core.txt"
    print(f"loading stock symbols from {pool_file} ...")
    stock_list = get_stock_list(pool_file)

    # extra_code = "600879" # 航天电子
    # stock_list.clear()
    # stock_list.append(extra_code)

  if not stock_list:
    print("💡 提示: 未发现有效股票代码，请检查 pool_core.txt 路径")
    return

  # [DEBUG]：仅测试前几条数据 ==========================================================
  # count = 10
  # print(f"🚀 开始测试模式：共 {len(stock_list)} 条，仅处理前 {count} 条。")
  # stock_list = stock_list[:count]
  # [DEBUG]：仅测试前几条数据 ==========================================================

  # 断点续传逻辑
  if start_code:
    try:
      stock_list = stock_list[stock_list.index(start_code):]
    except ValueError:
      pass

  pbar = tqdm(total=len(stock_list), leave=False, ncols=TQDM_NCOLS) if save else None
  lock = threading.Lock()

  with lock:
    for code in stock_list:
      try:
        # --- A. 自动补全 sh/sz 前缀: Ashare 的借口要求 ---
        if not (code.startswith('sh') or code.startswith('sz')):
          api_code = ('sh' + code) if code.startswith('6') else ('sz' + code)
        else:
          api_code = code

        # --- B. 安全调用接口 (绕过 Ashare 内部 DataFrame 构建错误) ---
        df = None
        try:
          # 尝试正常调用
          df = get_price(api_code, frequency='1d', count=1000)
        except Exception as e:
            message = f"{code} {exc}"
            if save and pbar is not None:
                tqdm.write(message)
                pbar.update(1)
            else:
                print(message)
            raise e

        if df is None or df.empty:
          if pbar: pbar.update(1)
          continue

        # --- C. 数据清洗与对齐 ---
        df = df.copy()
        df.reset_index(inplace=True)

        # Ashare/Sina 返回的列处理
        # 如果是手动抓取的，列名可能是 'day', 'open', 'high'...
        # 如果是 get_price 返回的，第一列可能是 'index'
        first_col = df.columns[0]
        rename_dict = {first_col: "trade_date", "volume": "vol"}
        df.rename(columns=rename_dict, inplace=True)

        # 注入代码并转换日期格式 (2021-01-01 -> 20210101)
        df.insert(0, "ts_code", code)
        df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.strftime("%Y%m%d")

        # --- D. 计算涨跌指标 (必须先按日期正序) ---
        df.sort_values(by="trade_date", ascending=True, inplace=True)
        df["change"] = df["close"].astype(float).diff()
        df["pct_change"] = df["close"].astype(float).pct_change() * 100

        # 重新转为降序 (最新在前)
        df.sort_values(by="trade_date", ascending=False, inplace=True)

        # 筛选 package 最终需要的列
        df = df.reindex(columns=[
          "ts_code", "trade_date", "open", "high", "low", "close",
          "change", "pct_change", "vol"
        ])

        # --- E. 保存文件 ---
        if save:
          save_path.mkdir(parents=True, exist_ok=True)
          # 确保保存时不带前缀的文件名
          df.to_csv(save_path / f"{code}.csv", index=False)
          if pbar: pbar.update(1)
        else:
          stock_data_queue.put(df)
          if len(stock_list) == 1:
            return df

      except Exception as exc:
        msg = f"❌ 股票 {code} 抓取失败: {exc}"
        if save and pbar:
          tqdm.write(msg)
          pbar.update(1)
        else:
          print(msg)
        continue

      # 稍微延长冷却，避免 IP 被封
      time.sleep(random.uniform(0.2, 0.5))

  if pbar: pbar.close()
  return None

def main() -> None:
    """Command-line entry point for fetching quote data."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--code", default="", type=str, help="single stock code or ticker")
    parser.add_argument("--api", default="akshare", type=str, help="api-provider: tushare, akshare or yfinance")
    parser.add_argument("--adjust", default="hfq", type=str, help="adjustment: none, qfq, or hfq")
    args = parser.parse_args()

    config.api = args.api
    config.adjust = args.adjust
    config.code = args.code

    if args.api == "yfinance":
        tickers = ["DAX", "IBM"]
        if not tickers:
            raise ValueError("Please provide at least one ticker when using yfinance")
        get_stock_data(tickers, save=True, save_path=daily_path)
        return

    if args.code:
        get_stock_data(args.code, save=True, save_path=daily_path)
    else:
        get_stock_data("", save=True, save_path=daily_path, datediff=-1)


if __name__ == "__main__":
    main()
