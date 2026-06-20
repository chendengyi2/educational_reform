"""蓝图注册"""
from routes.main import main_bp
from routes.algorithm_route import algorithm_bp
from routes.case_study_route import case_bp
from routes.auth_route import auth_bp


def register_blueprints(app):
    """注册所有蓝图"""
    app.register_blueprint(main_bp)
    app.register_blueprint(algorithm_bp, url_prefix='/algorithm')
    app.register_blueprint(case_bp, url_prefix='/case')
    app.register_blueprint(auth_bp, url_prefix='/auth')
