"""
财经大数据可视化分析平台 - Flask主应用
生产环境配置：将SQLite替换为MySQL，将内存缓存替换为Redis
"""
import os
import sys
import json
import hashlib
from datetime import datetime

# 确保项目根目录在Python路径中
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, BASE_DIR)

from flask import Flask, session
from models.models import db, User, AlgorithmModel, AnalysisResult, CaseStudy
from routes import register_blueprints


class MemoryCache:
    """内存字典模拟Redis缓存
    生产环境替换为：redis_client = redis.Redis(host='localhost', port=6379, db=0)
    """
    def __init__(self):
        self._store = {}

    def get(self, key):
        return self._store.get(key)

    def set(self, key, value, ex=None):
        self._store[key] = value

    def delete(self, key):
        self._store.pop(key, None)

    def exists(self, key):
        return key in self._store


cache = MemoryCache()


def create_app():
    """创建Flask应用"""
    app = Flask(
        __name__,
        template_folder=os.path.join(BASE_DIR, 'templates'),
        static_folder=os.path.join(BASE_DIR, 'static')
    )

    # 配置
    app.config['SECRET_KEY'] = 'finance-bigdata-platform-secret-key-2025'

    # SQLite数据库配置（保证可运行）
    # 生产环境替换为MySQL：
    # app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://user:password@localhost:3306/finance_bigdata'
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(BASE_DIR, 'data', 'finance.db')
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

    # 初始化数据库
    db.init_app(app)

    # 注册蓝图
    register_blueprints(app)

    # 创建数据库和初始数据
    with app.app_context():
        _init_database()

    # 模板上下文处理器
    @app.context_processor
    def inject_globals():
        return {
            'current_year': datetime.now().year,
            'app_name': '财经大数据可视化分析平台',
            'username': session.get('username', ''),
            'user_role': session.get('role', '')
        }

    return app


def _init_database():
    """初始化数据库和种子数据"""
    os.makedirs(os.path.join(BASE_DIR, 'data'), exist_ok=True)
    db.create_all()

    # 初始化管理员用户
    if User.query.filter_by(username='admin').first() is None:
        admin = User(
            username='admin',
            password_hash=hashlib.sha256('admin123'.encode('utf-8')).hexdigest(),
            role='admin',
            email='admin@finance-bigdata.com'
        )
        db.session.add(admin)

        analyst = User(
            username='analyst',
            password_hash=hashlib.sha256('analyst123'.encode('utf-8')).hexdigest(),
            role='analyst',
            email='analyst@finance-bigdata.com'
        )
        db.session.add(analyst)

        viewer = User(
            username='viewer',
            password_hash=hashlib.sha256('viewer123'.encode('utf-8')).hexdigest(),
            role='viewer',
            email='viewer@finance-bigdata.com'
        )
        db.session.add(viewer)
        db.session.commit()

    # 初始化算法模型
    if AlgorithmModel.query.count() == 0:
        seed_models = [
            {'name': '逻辑回归分类器', 'category': 'classification', 'description': '基于逻辑回归的二分类/多分类模型，适用于信贷审批、客户分群等场景', 'version': '2.1', 'status': 'deployed', 'accuracy': 0.872, 'f1_score': 0.851},
            {'name': '随机森林分类器', 'category': 'classification', 'description': '集成学习方法，通过构建多棵决策树提升分类准确率和鲁棒性', 'version': '3.0', 'status': 'deployed', 'accuracy': 0.915, 'f1_score': 0.893},
            {'name': 'XGBoost分类器', 'category': 'classification', 'description': '梯度提升树算法，在结构化数据上表现优异', 'version': '2.3', 'status': 'trained', 'accuracy': 0.934, 'f1_score': 0.921},
            {'name': 'K-Means聚类', 'category': 'clustering', 'description': '基于距离的聚类算法，用于客户分群、市场细分等', 'version': '1.5', 'status': 'deployed', 'accuracy': None, 'f1_score': None},
            {'name': 'DBSCAN聚类', 'category': 'clustering', 'description': '基于密度的聚类算法，能够发现任意形状的簇', 'version': '1.2', 'status': 'trained', 'accuracy': None, 'f1_score': None},
            {'name': '线性回归模型', 'category': 'regression', 'description': '经典线性回归，用于收益预测、风险评估等连续值预测', 'version': '2.0', 'status': 'deployed', 'accuracy': 0.88, 'f1_score': None},
            {'name': 'LSTM时序预测', 'category': 'regression', 'description': '长短期记忆网络，适用于股价预测、趋势分析等时序任务', 'version': '1.8', 'status': 'trained', 'accuracy': 0.91, 'f1_score': None},
            {'name': '协同过滤推荐', 'category': 'recommendation', 'description': '基于用户行为的协同过滤推荐引擎', 'version': '2.5', 'status': 'deployed', 'accuracy': 0.82, 'f1_score': None},
            {'name': '深度学习推荐模型', 'category': 'recommendation', 'description': '基于神经网络的推荐系统，融合多源特征', 'version': '1.3', 'status': 'training', 'accuracy': None, 'f1_score': None},
            {'name': 'ARIMA时序分析', 'category': 'timeseries', 'description': '自回归积分滑动平均模型，用于金融时间序列分析与预测', 'version': '3.1', 'status': 'deployed', 'accuracy': 0.86, 'f1_score': None},
            {'name': 'Prophet时序预测', 'category': 'timeseries', 'description': 'Facebook开源的时序预测工具，自动识别趋势和季节性', 'version': '1.0', 'status': 'trained', 'accuracy': 0.84, 'f1_score': None},
            {'name': 'DQN强化学习', 'category': 'reinforcement', 'description': '深度Q网络，用于动态资产配置和交易策略优化', 'version': '0.9', 'status': 'draft', 'accuracy': None, 'f1_score': None},
        ]
        for m in seed_models:
            model = AlgorithmModel(**m)
            db.session.add(model)
        db.session.commit()

    # 初始化分析结果
    if AnalysisResult.query.count() == 0:
        seed_results = [
            {
                'title': '信贷风控模型准确率对比',
                'chart_type': 'bar',
                'data_json': json.dumps({
                    'categories': ['逻辑回归', '决策树', '随机森林', 'XGBoost', 'LightGBM'],
                    'values': [0.85, 0.82, 0.91, 0.93, 0.92]
                }, ensure_ascii=False),
                'summary': '对比了5种分类算法在信贷风控场景的准确率表现，XGBoost以93%的准确率领先。'
            },
            {
                'title': '客户价值聚类分析',
                'chart_type': 'scatter',
                'data_json': json.dumps({'type': 'cluster_scatter'}, ensure_ascii=False),
                'summary': '通过K-Means聚类将客户分为3类：高价值客户(25%)、中等价值客户(45%)、低价值客户(30%)。'
            },
            {
                'title': '股价走势预测',
                'chart_type': 'line',
                'data_json': json.dumps({
                    'dates': ['1月', '2月', '3月', '4月', '5月', '6月'],
                    'actual': [3200, 3350, 3180, 3420, 3510, 3680],
                    'predicted': [3220, 3300, 3250, 3380, 3480, 3650]
                }, ensure_ascii=False),
                'summary': 'LSTM模型对未来6个月股价走势的预测，平均误差率为3.2%。'
            },
            {
                'title': '金融产品推荐占比',
                'chart_type': 'pie',
                'data_json': json.dumps([
                    {'name': '基金理财', 'value': 35},
                    {'name': '保险产品', 'value': 25},
                    {'name': '贷款服务', 'value': 20},
                    {'name': '信用卡', 'value': 12},
                    {'name': '外汇服务', 'value': 8}
                ], ensure_ascii=False),
                'summary': '协同过滤推荐系统的产品推荐分布，基金理财占比最高。'
            }
        ]
        for r in seed_results:
            result = AnalysisResult(**r)
            db.session.add(result)
        db.session.commit()


app = create_app()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
