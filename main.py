"""
财经大数据教改项目 - 总入口

四大模块：
  模块一：财经大数据收集与预处理
  模块二：财经大数据分析与挖掘
  模块三：案例资源库建设
  模块四：可视化平台开发

使用方式：
  python main.py --module1 --demo    # 运行模块一演示
  python main.py --module2 --demo    # 运行模块二演示
  python main.py --module3 --list    # 列出所有案例
  python main.py --module4 --port 5000  # 启动可视化平台
"""

import sys
import subprocess
import argparse
from pathlib import Path


def run_module(script, extra_args=None):
    cmd = [sys.executable, str(script)]
    if extra_args:
        cmd.extend(extra_args)
    subprocess.run(cmd, check=False)


def main():
    parser = argparse.ArgumentParser(description="财经大数据教改项目")
    parser.add_argument("--module1", action="store_true", help="模块一：财经大数据收集与预处理")
    parser.add_argument("--module2", action="store_true", help="模块二：财经大数据分析与挖掘")
    parser.add_argument("--module3", action="store_true", help="模块三：案例资源库建设")
    parser.add_argument("--module4", action="store_true", help="模块四：可视化平台开发")
    parser.add_argument("--demo", action="store_true", help="运行演示")
    parser.add_argument("--list", action="store_true", help="列出案例")
    parser.add_argument("--port", type=int, default=5000, help="平台端口")
    parser.add_argument("--run", type=int, help="运行指定案例(1-8)")
    args = parser.parse_args()

    if args.module1:
        extra = ["--demo"] if args.demo else []
        run_module("run_module1.py", extra)
    elif args.module2:
        extra = ["--demo"] if args.demo else []
        run_module("run_module2.py", extra)
    elif args.module3:
        extra = ["--list"] if args.list else (["--run", str(args.run)] if args.run else [])
        run_module("run_module3.py", extra)
    elif args.module4:
        run_module("run_module4.py", ["--port", str(args.port)])
    else:
        print("""
财经大数据教改项目 - 四大模块
================================

  模块一：财经大数据收集与预处理
    python main.py --module1 --demo

  模块二：财经大数据分析与挖掘
    python main.py --module2 --demo

  模块三：案例资源库建设
    python main.py --module3 --list          # 列出8个案例
    python main.py --module3 --run 1         # 运行案例1

  模块四：可视化平台开发
    python main.py --module4 --port 5000     # 启动平台
""")


if __name__ == "__main__":
    main()
