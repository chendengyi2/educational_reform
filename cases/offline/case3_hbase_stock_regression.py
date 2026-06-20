#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
案例3：基于HBase的股票行情数据存储与回归预测

模拟HBase的KV存储结构，使用字典模拟行键+列族，
完成股票行情特征工程与多回归模型预测对比。

大数据技术栈说明：
- 实际环境：股票行情数据通过Flume采集写入HBase，行键为 股票代码+时间戳倒序
- 本地模拟：使用Python字典模拟HBase的RowKey+ColumnFamily结构
- 数据规模：实际HBase存储千万级时间序列数据，此处模拟5只股票各500个交易日
- 读取方式：实际通过HBase Scan操作批量读取，此处用字典遍历模拟
"""

import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error
import warnings

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================
# 第一部分：模拟HBase存储结构
# HBase数据模型：RowKey -> ColumnFamily:Qualifier -> Value
# 行键设计：stock_code + reverse_timestamp（倒序，最新数据在前）
# 列族设计：cf:quote（行情列族）、cf:indicator（指标列族）
# ============================================================

class HBaseSimulator:
    """
    模拟HBase的KV存储结构
    实际HBase操作：
      - Put: htable.put( Put(rowKey).addColumn(cf, qual, value) )
      - Scan: htable.getScanner( Scan().setStartRow(start).setStopRow(stop) )
    """

    def __init__(self, table_name):
        self.table_name = table_name
        self.data = {}  # {row_key: {cf:qual: value}}
        logger.info(f"[HBase] 创建表: {table_name}")

    def put(self, row_key, column_family, qualifier, value):
        """模拟HBase Put操作"""
        if row_key not in self.data:
            self.data[row_key] = {}
        col_key = f"{column_family}:{qualifier}"
        self.data[row_key][col_key] = value

    def scan(self, prefix=None, limit=None):
        """
        模拟HBase Scan操作
        实际环境：Scan扫描指定RowKey范围的数据
        """
        results = []
        sorted_keys = sorted(self.data.keys())
        for rk in sorted_keys:
            if prefix and not rk.startswith(prefix):
                continue
            results.append((rk, self.data[rk]))
            if limit and len(results) >= limit:
                break
        return results

    def get(self, row_key):
        """模拟HBase Get操作"""
        return self.data.get(row_key, {})


def generate_stock_data_to_hbase(n_stocks=5, n_days=500):
    """
    生成股票行情数据并写入模拟HBase
    实际环境：数据通过Flume实时采集写入HBase
    """
    np.random.seed(42)
    stock_codes = ["SH600519", "SH601318", "SZ000858", "SZ000333", "SH600036"]
    stock_names = ["贵州茅台", "中国平安", "五粮液", "美的集团", "招商银行"]

    hbase_table = HBaseSimulator("stock:daily_quote")

    base_prices = [1800.0, 75.0, 220.0, 65.0, 38.0]
    base_timestamp = 1700000000  # 模拟时间戳起点

    for s_idx, (code, name) in enumerate(zip(stock_codes, stock_names)):
        price = base_prices[s_idx]
        for d in range(n_days):
            # 行键：股票代码 + 时间戳倒序（HBase设计，最新数据排在前面）
            reverse_ts = str(9999999999 - (base_timestamp + d * 86400))
            row_key = f"{code}#{reverse_ts}"

            # 模拟股价走势（几何布朗运动）
            daily_return = np.random.normal(0.0003, 0.02)
            price = price * np.exp(daily_return)
            high = price * (1 + abs(np.random.normal(0, 0.01)))
            low = price * (1 - abs(np.random.normal(0, 0.01)))
            open_price = price * (1 + np.random.normal(0, 0.005))
            volume = int(np.random.lognormal(14, 1.0))

            # 写入行情列族 cf:quote
            hbase_table.put(row_key, "quote", "open", round(open_price, 2))
            hbase_table.put(row_key, "quote", "high", round(high, 2))
            hbase_table.put(row_key, "quote", "low", round(low, 2))
            hbase_table.put(row_key, "quote", "close", round(price, 2))
            hbase_table.put(row_key, "quote", "volume", volume)

    logger.info(f"[HBase] 写入完成，总行数: {len(hbase_table.data)}")
    return hbase_table, stock_codes, stock_names, n_days


# ============================================================
# 第二部分：从HBase读取数据并转换为DataFrame
# 实际环境：通过HBase Scan读取后转为Spark DataFrame
# ============================================================

def hbase_scan_to_dataframe(hbase_table, stock_codes, n_days):
    """
    模拟HBase Scan操作，将数据转为pandas DataFrame
    等价于Spark读取HBase后创建DataFrame
    """
    all_records = []

    for code in stock_codes:
        # Scan该股票的所有数据
        scan_results = hbase_table.scan(prefix=code)
        for row_key, cols in scan_results:
            record = {
                "stock_code": code,
                "date_idx": int(row_key.split("#")[1]),
                "open": cols.get("quote:open", 0),
                "high": cols.get("quote:high", 0),
                "low": cols.get("quote:low", 0),
                "close": cols.get("quote:close", 0),
                "volume": cols.get("quote:volume", 0),
            }
            all_records.append(record)

    df = pd.DataFrame(all_records)
    df = df.sort_values(["stock_code", "date_idx"]).reset_index(drop=True)
    logger.info(f"[HBase Scan] 读取完成，总记录数: {len(df)}")
    return df


# ============================================================
# 第三部分：技术指标特征工程
# ============================================================

def compute_technical_indicators(df):
    """
    计算技术指标特征：MA、RSI、MACD
    实际环境：通过Spark UDF或Hive UDF计算
    """
    result_frames = []

    for code in df["stock_code"].unique():
        stock_df = df[df["stock_code"] == code].copy().reset_index(drop=True)

        close = stock_df["close"]
        volume = stock_df["volume"]

        # 滞后特征
        for lag in [1, 2, 3, 5]:
            stock_df[f"close_lag_{lag}"] = close.shift(lag)
            stock_df[f"volume_lag_{lag}"] = volume.shift(lag)

        # 滚动统计特征
        for window in [5, 10, 20]:
            stock_df[f"close_ma_{window}"] = close.rolling(window).mean()
            stock_df[f"close_std_{window}"] = close.rolling(window).std()
            stock_df[f"volume_ma_{window}"] = volume.rolling(window).mean()

        # 移动平均线 (MA)
        stock_df["ma5"] = close.rolling(5).mean()
        stock_df["ma10"] = close.rolling(10).mean()
        stock_df["ma20"] = close.rolling(20).mean()
        stock_df["ma60"] = close.rolling(60).mean()

        # RSI (14日)
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, 1e-10)
        stock_df["rsi_14"] = (100 - 100 / (1 + rs)).round(2)

        # MACD
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        stock_df["macd_dif"] = (ema12 - ema26).round(4)
        stock_df["macd_dea"] = stock_df["macd_dif"].ewm(span=9, adjust=False).mean().round(4)
        stock_df["macd_hist"] = (stock_df["macd_dif"] - stock_df["macd_dea"]).round(4)

        # 价格变动特征
        stock_df["pct_change"] = close.pct_change()
        stock_df["high_low_ratio"] = stock_df["high"] / stock_df["low"]
        stock_df["close_open_ratio"] = close / stock_df["open"]

        # 目标变量：下一日收盘价
        stock_df["target_next_close"] = close.shift(-1)

        result_frames.append(stock_df)

    merged = pd.concat(result_frames, ignore_index=True)
    merged = merged.dropna().reset_index(drop=True)
    logger.info(f"[特征工程] 技术指标计算完成，有效记录数: {len(merged)}")
    return merged


# ============================================================
# 第四部分：回归模型训练与对比
# ============================================================

def train_and_compare_regression(df):
    """训练线性回归、岭回归、Lasso、GBDT并对比"""
    feature_cols = [
        "close", "volume", "close_lag_1", "close_lag_2", "close_lag_3", "close_lag_5",
        "volume_lag_1", "volume_lag_5",
        "close_ma_5", "close_ma_10", "close_ma_20",
        "close_std_5", "close_std_10",
        "volume_ma_5", "volume_ma_10",
        "ma5", "ma10", "ma20",
        "rsi_14", "macd_dif", "macd_dea", "macd_hist",
        "pct_change", "high_low_ratio", "close_open_ratio",
    ]

    X = df[feature_cols].values
    y = df["target_next_close"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    logger.info(f"训练集: {len(X_train)}, 测试集: {len(X_test)}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "线性回归": LinearRegression(),
        "岭回归": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.01, max_iter=5000),
        "GBDT": GradientBoostingRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42),
    }

    results = {}
    for name, model in models.items():
        if name in ["线性回归", "岭回归", "Lasso"]:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        results[name] = {"r2": r2, "rmse": rmse, "predictions": y_pred}
        logger.info(f"  {name}: R2={r2:.4f}, RMSE={rmse:.4f}")

    return results, y_test, feature_cols


# ============================================================
# 第五部分：结果输出
# ============================================================

def print_results(results, y_test, stock_names):
    """输出完整的分析结果"""
    print("\n" + "=" * 70)
    print("  基于HBase的股票行情数据存储与回归预测 - 结果报告")
    print("=" * 70)

    # 模型对比
    print("\n【一】回归模型性能对比")
    print("-" * 45)
    print(f"{'模型':<12} {'R2':<12} {'RMSE':<12}")
    print("-" * 45)
    for name, res in results.items():
        print(f"{name:<12} {res['r2']:<12.4f} {res['rmse']:<12.4f}")

    # 预测示例
    print("\n【二】各模型预测示例（前5个样本）")
    print("-" * 70)
    print(f"{'真实值':<12}", end="")
    for name in results:
        print(f"{name+'预测':<14}", end="")
    print()
    print("-" * 70)
    for i in range(min(5, len(y_test))):
        print(f"{y_test[i]:<12.2f}", end="")
        for name in results:
            print(f"{results[name]['predictions'][i]:<14.2f}", end="")
        print()

    # 最优模型
    best_name = max(results, key=lambda x: results[x]["r2"])
    print(f"\n【三】最优模型: {best_name} (R2={results[best_name]['r2']:.4f})")

    # HBase存储说明
    print("\n【四】HBase存储设计说明")
    print("-" * 55)
    print("  行键设计: 股票代码#倒序时间戳 (如 SH600519#9999998369)")
    print("  列族设计: quote(行情) / indicator(技术指标)")
    print("  Scan操作: 按股票代码前缀扫描，倒序时间戳保证最新数据优先")
    print("  数据量:   5只股票 × 500交易日 = 2,500行记录")
    print(f"  股票列表: {', '.join(stock_names)}")

    print("\n" + "=" * 70)


# ============================================================
# 主函数
# ============================================================

def main():
    logger.info("=" * 50)
    logger.info("开始执行：基于HBase的股票行情数据存储与回归预测")
    logger.info("=" * 50)

    # Step 1: 生成数据并写入HBase
    logger.info("\n--- Step 1: 生成股票数据并写入HBase ---")
    hbase_table, stock_codes, stock_names, n_days = generate_stock_data_to_hbase()

    # Step 2: 从HBase Scan读取数据
    logger.info("\n--- Step 2: 从HBase Scan读取数据 ---")
    df = hbase_scan_to_dataframe(hbase_table, stock_codes, n_days)

    # Step 3: 技术指标特征工程
    logger.info("\n--- Step 3: 计算技术指标特征 ---")
    df_features = compute_technical_indicators(df)

    # Step 4: 回归模型训练与对比
    logger.info("\n--- Step 4: 回归模型训练与对比 ---")
    results, y_test, feature_cols = train_and_compare_regression(df_features)

    # Step 5: 结果输出
    print_results(results, y_test, stock_names)

    logger.info("分析完成！")


if __name__ == "__main__":
    main()
