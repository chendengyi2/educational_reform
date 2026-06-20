"""
模块四：可视化平台开发 - 启动程序

使用方式：
  python run_module4.py              # 启动平台(默认5000端口)
  python run_module4.py --port 8080  # 指定端口

访问地址：
  http://localhost:5000              # 首页
  http://localhost:5000/dashboard    # 数据概览大屏
  http://localhost:5000/algorithm/   # 算法模型管理
  http://localhost:5000/case/        # 案例演示
  http://localhost:5000/visualization # 结果可视化
  http://localhost:5000/auth/login   # 登录页

默认账号：
  admin / admin123 (管理员)
  analyst / analyst123 (分析师)
  viewer / viewer123 (查看者)

技术栈：
  后端：Flask + PyMySQL + SQLAlchemy（开发环境用SQLite）
  前端：Echarts + Bootstrap5 + jQuery
  数据层：MySQL（结果存储）+ Redis（缓存，开发环境用内存字典）
"""

import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / "module4_platform"))

from app import app


def main():
    parser = argparse.ArgumentParser(description="模块四：可视化平台开发")
    parser.add_argument("--port", type=int, default=5000, help="端口号")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="绑定地址")
    args = parser.parse_args()

    print("\n" + "=" * 50)
    print("  财经大数据可视化分析平台")
    print("=" * 50)
    print(f"  访问地址: http://localhost:{args.port}")
    print(f"  数据大屏: http://localhost:{args.port}/dashboard")
    print(f"  默认账号: admin / admin123")
    print("=" * 50 + "\n")

    app.run(host=args.host, port=args.port, debug=True)


if __name__ == "__main__":
    main()
