"""主页和概览大屏路由"""
import json
from flask import Blueprint, render_template, jsonify
from models.models import db, AlgorithmModel, AnalysisResult, CaseStudy, User

main_bp = Blueprint('main', __name__)


@main_bp.route('/')
def index():
    """首页重定向到大屏"""
    return render_template('dashboard.html')


@main_bp.route('/dashboard')
def dashboard():
    """数据概览大屏"""
    return render_template('dashboard.html')


@main_bp.route('/visualization')
def visualization():
    """结果可视化页面"""
    return render_template('visualization.html')


@main_bp.route('/api/overview')
def api_overview():
    """概览数据API - 供大屏页面调用"""
    # 统计数据
    model_count = AlgorithmModel.query.count()
    result_count = AnalysisResult.query.count()
    user_count = User.query.count()

    try:
        case_count = CaseStudy.query.count()
    except Exception:
        case_count = 8

    return jsonify({
        'stats': {
            'model_count': model_count,
            'result_count': result_count,
            'case_count': case_count,
            'user_count': user_count
        }
    })


@main_bp.route('/api/dashboard/charts')
def api_dashboard_charts():
    """大屏图表数据API"""
    # 分类算法对比柱状图数据
    bar_data = {
        'categories': ['逻辑回归', '决策树', '随机森林', 'SVM', 'XGBoost', '神经网络', 'KNN', '朴素贝叶斯'],
        'values': {
            '准确率': [0.85, 0.82, 0.91, 0.87, 0.93, 0.90, 0.79, 0.81],
            'F1分数': [0.83, 0.80, 0.89, 0.85, 0.92, 0.88, 0.77, 0.79],
            '召回率': [0.81, 0.78, 0.88, 0.84, 0.91, 0.87, 0.75, 0.77]
        }
    }

    # 聚类散点图数据
    cluster_data = _generate_cluster_data()

    # 回归预测折线图数据
    regression_data = {
        'dates': ['2025-01', '2025-02', '2025-03', '2025-04', '2025-05', '2025-06',
                  '2025-07', '2025-08', '2025-09', '2025-10', '2025-11', '2025-12'],
        'actual': [2340, 2510, 2380, 2650, 2780, 2890, 3020, 3150, 2980, 3210, 3350, 3480],
        'predicted': [2300, 2480, 2420, 2600, 2750, 2920, 3050, 3120, 3010, 3180, 3320, 3510]
    }

    # 推荐系统饼图数据
    pie_data = [
        {'name': '协同过滤', 'value': 35},
        {'name': '内容推荐', 'value': 25},
        {'name': '深度学习推荐', 'value': 20},
        {'name': '混合推荐', 'value': 12},
        {'name': '知识图谱推荐', 'value': 8}
    ]

    # 风险评估雷达图数据
    radar_data = {
        'indicators': [
            {'name': '市场风险', 'max': 100},
            {'name': '信用风险', 'max': 100},
            {'name': '操作风险', 'max': 100},
            {'name': '流动性风险', 'max': 100},
            {'name': '合规风险', 'max': 100}
        ],
        'series': [
            {'name': '当前评分', 'value': [72, 85, 68, 76, 90]},
            {'name': '行业均值', 'value': [65, 78, 72, 70, 82]}
        ]
    }

    return jsonify({
        'bar': bar_data,
        'cluster': cluster_data,
        'regression': regression_data,
        'pie': pie_data,
        'radar': radar_data
    })


def _generate_cluster_data():
    """生成聚类散点图模拟数据"""
    import random
    random.seed(42)
    clusters = {
        '低风险客户': {'center': (2, 8), 'color': '#5470c6', 'count': 30},
        '中等风险客户': {'center': (5, 5), 'color': '#91cc75', 'count': 40},
        '高风险客户': {'center': (8, 2), 'color': '#ee6666', 'count': 25}
    }
    data = []
    for name, info in clusters.items():
        for _ in range(info['count']):
            x = info['center'][0] + random.uniform(-1.5, 1.5)
            y = info['center'][1] + random.uniform(-1.5, 1.5)
            data.append([round(x, 2), round(y, 2), name])
    return data


@main_bp.route('/api/visualization/data')
def api_visualization_data():
    """结果可视化页面数据API"""
    # 热力图数据 - 资产相关性矩阵
    assets = ['股票', '债券', '基金', '期货', '外汇', '数字货币']
    heatmap_data = []
    for i in range(len(assets)):
        for j in range(len(assets)):
            if i == j:
                val = 1.0
            elif j > i:
                import random
                random.seed(i * 10 + j)
                val = round(random.uniform(-0.5, 0.9), 2)
                heatmap_data.append([i, j, val])
                heatmap_data.append([j, i, val])
            # symmetric - skip when j < i (already added)

    # 时序分析数据
    import random
    random.seed(99)
    time_series = {
        'dates': [f'2025-{str(m).zfill(2)}-{str(d).zfill(2)}' for m in range(1, 7) for d in range(1, 29, 7)],
        'series': {
            '沪深300': [round(3800 + random.uniform(-200, 200), 2) for _ in range(24)],
            '中证500': [round(6200 + random.uniform(-300, 300), 2) for _ in range(24)],
            '创业板指': [round(2100 + random.uniform(-150, 150), 2) for _ in range(24)]
        }
    }

    return jsonify({
        'heatmap': {'assets': assets, 'data': heatmap_data},
        'time_series': time_series
    })
