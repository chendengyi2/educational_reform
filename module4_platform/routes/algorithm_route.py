"""算法模型管理路由"""
import json
from flask import Blueprint, render_template, jsonify, request, session
from models.models import db, AlgorithmModel

algorithm_bp = Blueprint('algorithm', __name__)


@algorithm_bp.route('/')
def index():
    """算法管理页面"""
    return render_template('algorithm.html')


@algorithm_bp.route('/api/list')
def api_list():
    """获取算法列表"""
    category = request.args.get('category', '')
    status = request.args.get('status', '')
    keyword = request.args.get('keyword', '')

    query = AlgorithmModel.query
    if category:
        query = query.filter_by(category=category)
    if status:
        query = query.filter_by(status=status)
    if keyword:
        query = query.filter(AlgorithmModel.name.contains(keyword))

    models = query.order_by(AlgorithmModel.updated_at.desc()).all()
    return jsonify({
        'code': 0,
        'data': [m.to_dict() for m in models],
        'total': len(models)
    })


@algorithm_bp.route('/api/detail/<int:model_id>')
def api_detail(model_id):
    """获取算法详情"""
    model = AlgorithmModel.query.get_or_404(model_id)
    return jsonify({'code': 0, 'data': model.to_dict()})


@algorithm_bp.route('/api/create', methods=['POST'])
def api_create():
    """创建算法模型"""
    data = request.get_json()
    model = AlgorithmModel(
        name=data.get('name', ''),
        category=data.get('category', ''),
        description=data.get('description', ''),
        version=data.get('version', '1.0'),
        params=json.dumps(data.get('params', {}), ensure_ascii=False),
        status='draft'
    )
    db.session.add(model)
    db.session.commit()
    return jsonify({'code': 0, 'msg': '创建成功', 'data': model.to_dict()})


@algorithm_bp.route('/api/train/<int:model_id>', methods=['POST'])
def api_train(model_id):
    """触发模型训练（模拟）"""
    model = AlgorithmModel.query.get_or_404(model_id)
    model.status = 'training'
    db.session.commit()

    # 模拟训练完成
    import random
    random.seed(model_id)
    model.status = 'trained'
    model.accuracy = round(random.uniform(0.80, 0.96), 4)
    model.f1_score = round(random.uniform(0.78, 0.94), 4)
    db.session.commit()

    return jsonify({
        'code': 0,
        'msg': '训练完成',
        'data': model.to_dict()
    })


@algorithm_bp.route('/api/deploy/<int:model_id>', methods=['POST'])
def api_deploy(model_id):
    """部署模型"""
    model = AlgorithmModel.query.get_or_404(model_id)
    if model.status != 'trained':
        return jsonify({'code': 1, 'msg': '只有已训练的模型才能部署'})
    model.status = 'deployed'
    db.session.commit()
    return jsonify({'code': 0, 'msg': '部署成功', 'data': model.to_dict()})


@algorithm_bp.route('/api/delete/<int:model_id>', methods=['DELETE'])
def api_delete(model_id):
    """删除模型"""
    role = session.get('role', 'viewer')
    if role != 'admin':
        return jsonify({'code': 1, 'msg': '权限不足，仅管理员可删除'})

    model = AlgorithmModel.query.get_or_404(model_id)
    db.session.delete(model)
    db.session.commit()
    return jsonify({'code': 0, 'msg': '删除成功'})


@algorithm_bp.route('/api/categories')
def api_categories():
    """获取算法分类列表"""
    categories = [
        {'key': 'classification', 'name': '分类算法', 'icon': 'bi-diagram-3'},
        {'key': 'clustering', 'name': '聚类算法', 'icon': 'bi-circle-square'},
        {'key': 'regression', 'name': '回归算法', 'icon': 'bi-graph-up-arrow'},
        {'key': 'recommendation', 'name': '推荐算法', 'icon': 'bi-hand-thumbs-up'},
        {'key': 'timeseries', 'name': '时序分析', 'icon': 'bi-clock-history'},
        {'key': 'reinforcement', 'name': '强化学习', 'icon': 'bi-robot'}
    ]
    return jsonify({'code': 0, 'data': categories})
