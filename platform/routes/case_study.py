"""案例演示路由"""
from flask import Blueprint, render_template, jsonify, request
from models.models import db, CaseStudy

case_bp = Blueprint('case', __name__)

# 8个案例的默认数据
DEFAULT_CASES = [
    {
        'id': 1, 'title': '信贷风险评估案例', 'category': '风控',
        'description': '基于机器学习算法对个人信贷申请进行风险评估，自动识别高风险客户，降低坏账率。',
        'algorithms': '逻辑回归,随机森林,XGBoost',
        'content': '本案例使用某银行3年信贷数据，包含50万条贷款记录。通过特征工程提取了200+维度特征，使用逻辑回归、随机森林和XGBoost构建风险评分模型。最终模型AUC达到0.92，坏账识别率提升35%。'
    },
    {
        'id': 2, 'title': '智能营销推荐案例', 'category': '营销',
        'description': '利用协同过滤和深度学习推荐算法，实现金融产品精准推荐，提升交叉销售转化率。',
        'algorithms': '协同过滤,深度学习推荐,混合推荐',
        'content': '基于用户行为数据和交易数据构建推荐系统，采用协同过滤捕获用户偏好，结合深度学习模型挖掘隐含特征。A/B测试显示推荐点击率提升48%，交叉销售转化率提升22%。'
    },
    {
        'id': 3, 'title': '量化投资策略案例', 'category': '投资',
        'description': '运用多因子模型和机器学习方法构建量化选股策略，实现超额收益。',
        'algorithms': '多因子模型,LSTM,强化学习',
        'content': '本案例基于A股市场10年历史数据，构建了包含价值、成长、质量、动量四大类共50个因子的多因子模型。结合LSTM预测股价走势和强化学习动态调仓，回测年化收益达23.5%，夏普比1.8。'
    },
    {
        'id': 4, 'title': '信用评分模型案例', 'category': '信用',
        'description': '构建个人信用评分体系，为金融机构提供客户信用等级评定依据。',
        'algorithms': '逻辑回归,LightGBM,评分卡模型',
        'content': '整合央行征信数据、社交数据和行为数据，使用WOE编码和IV值筛选特征，构建信用评分卡模型。模型KS值达0.45，区分度优良，已为200万+用户完成信用评分。'
    },
    {
        'id': 5, 'title': '反欺诈检测案例', 'category': '反欺诈',
        'description': '实时检测金融交易中的欺诈行为，保护用户资金安全。',
        'algorithms': '孤立森林,深度学习,图神经网络',
        'content': '基于交易流水和设备指纹数据，构建实时反欺诈系统。使用孤立森林进行异常检测，图神经网络识别欺诈团伙，深度学习模型实时评分。系统准确率达99.2%，日均处理交易1000万笔，欺诈损失降低60%。'
    },
    {
        'id': 6, 'title': '高频交易分析案例', 'category': '量化',
        'description': '分析高频交易数据特征，识别市场微观结构模式，辅助交易决策。',
        'algorithms': '时序分析,深度学习,强化学习',
        'content': '对Level-2逐笔数据进行特征提取，包括订单簿不平衡度、成交量加权平均价偏离等微观结构特征。使用Transformer模型预测短期价格变动，策略回测胜率达58%，日均收益稳定。'
    },
    {
        'id': 7, 'title': '舆情分析预警案例', 'category': '舆情',
        'description': '实时采集分析金融新闻和社交媒体舆情，预警市场情绪变化和潜在风险。',
        'algorithms': 'NLP,情感分析,知识图谱',
        'content': '构建金融领域知识图谱，整合新闻、研报、社交媒体等多源数据。使用BERT模型进行情感分析，结合知识图谱实现事件关联推理。系统可提前2小时预警负面舆情，准确率达87%。'
    },
    {
        'id': 8, 'title': '供应链金融风控案例', 'category': '供应链',
        'description': '利用图算法和大数据分析评估供应链金融风险，为中小企业融资提供决策支持。',
        'algorithms': '图算法,聚类分析,风险评估',
        'content': '整合工商、税务、司法、海关等多维数据，构建供应链企业关系图谱。使用社区发现算法识别核心企业圈，结合企业财务指标评估风险传导路径。模型成功预警3起供应链违约事件，为银行减少潜在损失2亿元。'
    }
]


@case_bp.route('/')
def index():
    """案例展示页面"""
    return render_template('cases.html')


@case_bp.route('/api/list')
def api_list():
    """获取案例列表"""
    category = request.args.get('category', '')

    # 尝试从数据库获取，为空则返回默认数据
    try:
        query = CaseStudy.query
        if category:
            query = query.filter_by(category=category)
        cases = query.all()
        if cases:
            return jsonify({'code': 0, 'data': [c.to_dict() for c in cases]})
    except Exception:
        pass

    # 返回默认案例数据
    filtered = DEFAULT_CASES
    if category:
        filtered = [c for c in DEFAULT_CASES if c['category'] == category]
    return jsonify({'code': 0, 'data': filtered})


@case_bp.route('/api/detail/<int:case_id>')
def api_detail(case_id):
    """获取案例详情"""
    try:
        case = CaseStudy.query.get(case_id)
        if case:
            return jsonify({'code': 0, 'data': case.to_dict()})
    except Exception:
        pass

    # 从默认数据中查找
    for case in DEFAULT_CASES:
        if case['id'] == case_id:
            return jsonify({'code': 0, 'data': case})
    return jsonify({'code': 1, 'msg': '案例不存在'}), 404


@case_bp.route('/api/categories')
def api_categories():
    """获取案例分类列表"""
    categories = [
        {'key': '风控', 'name': '风控管理', 'icon': 'bi-shield-check', 'count': 1},
        {'key': '营销', 'name': '智能营销', 'icon': 'bi-megaphone', 'count': 1},
        {'key': '投资', 'name': '量化投资', 'icon': 'bi-graph-up', 'count': 1},
        {'key': '信用', 'name': '信用评估', 'icon': 'bi-award', 'count': 1},
        {'key': '反欺诈', 'name': '反欺诈', 'icon': 'bi-exclamation-triangle', 'count': 1},
        {'key': '量化', 'name': '高频交易', 'icon': 'bi-lightning', 'count': 1},
        {'key': '舆情', 'name': '舆情分析', 'icon': 'bi-chat-dots', 'count': 1},
        {'key': '供应链', 'name': '供应链金融', 'icon': 'bi-link-45deg', 'count': 1}
    ]
    return jsonify({'code': 0, 'data': categories})
