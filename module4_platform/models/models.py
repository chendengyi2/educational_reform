"""
数据模型定义
生产环境：使用MySQL数据库，通过SQLAlchemy连接
"""
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


class User(db.Model):
    """用户模型"""
    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    role = db.Column(db.String(20), nullable=False, default='viewer')  # admin / analyst / viewer
    email = db.Column(db.String(120), unique=True, nullable=True)
    avatar = db.Column(db.String(256), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.now)
    last_login = db.Column(db.DateTime, nullable=True)
    is_active = db.Column(db.Boolean, default=True)

    def to_dict(self):
        return {
            'id': self.id,
            'username': self.username,
            'role': self.role,
            'email': self.email,
            'avatar': self.avatar,
            'created_at': self.created_at.strftime('%Y-%m-%d %H:%M:%S') if self.created_at else '',
            'last_login': self.last_login.strftime('%Y-%m-%d %H:%M:%S') if self.last_login else '',
            'is_active': self.is_active
        }


class AlgorithmModel(db.Model):
    """算法模型"""
    __tablename__ = 'algorithm_models'

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    name = db.Column(db.String(100), nullable=False)
    category = db.Column(db.String(50), nullable=False)  # 分类/聚类/回归/推荐/时序/强化学习
    description = db.Column(db.Text, nullable=True)
    version = db.Column(db.String(20), default='1.0')
    params = db.Column(db.Text, nullable=True)  # JSON格式参数
    accuracy = db.Column(db.Float, nullable=True)
    f1_score = db.Column(db.Float, nullable=True)
    status = db.Column(db.String(20), default='draft')  # draft/training/trained/deployed
    created_at = db.Column(db.DateTime, default=datetime.now)
    updated_at = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now)

    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'category': self.category,
            'description': self.description,
            'version': self.version,
            'params': self.params,
            'accuracy': self.accuracy,
            'f1_score': self.f1_score,
            'status': self.status,
            'created_at': self.created_at.strftime('%Y-%m-%d %H:%M:%S') if self.created_at else '',
            'updated_at': self.updated_at.strftime('%Y-%m-%d %H:%M:%S') if self.updated_at else ''
        }


class AnalysisResult(db.Model):
    """分析结果模型"""
    __tablename__ = 'analysis_results'

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    title = db.Column(db.String(200), nullable=False)
    algorithm_id = db.Column(db.Integer, db.ForeignKey('algorithm_models.id'), nullable=True)
    chart_type = db.Column(db.String(30), nullable=False)  # bar/line/pie/scatter/heatmap/radar
    data_json = db.Column(db.Text, nullable=False)  # JSON格式图表数据
    summary = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.now)

    algorithm = db.relationship('AlgorithmModel', backref=db.backref('results', lazy=True))

    def to_dict(self):
        return {
            'id': self.id,
            'title': self.title,
            'algorithm_id': self.algorithm_id,
            'chart_type': self.chart_type,
            'data_json': self.data_json,
            'summary': self.summary,
            'created_at': self.created_at.strftime('%Y-%m-%d %H:%M:%S') if self.created_at else ''
        }


class CaseStudy(db.Model):
    """案例研究模型"""
    __tablename__ = 'case_studies'

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    title = db.Column(db.String(200), nullable=False)
    category = db.Column(db.String(50), nullable=False)  # 风控/营销/投资/信用/反欺诈/量化/舆情/供应链
    description = db.Column(db.Text, nullable=True)
    algorithm_ids = db.Column(db.Text, nullable=True)  # 逗号分隔的算法ID
    cover_image = db.Column(db.String(256), nullable=True)
    content = db.Column(db.Text, nullable=True)
    status = db.Column(db.String(20), default='published')  # draft/published
    views = db.Column(db.Integer, default=0)
    created_at = db.Column(db.DateTime, default=datetime.now)

    def to_dict(self):
        return {
            'id': self.id,
            'title': self.title,
            'category': self.category,
            'description': self.description,
            'algorithm_ids': self.algorithm_ids,
            'cover_image': self.cover_image,
            'content': self.content,
            'status': self.status,
            'views': self.views,
            'created_at': self.created_at.strftime('%Y-%m-%d %H:%M:%S') if self.created_at else ''
        }
