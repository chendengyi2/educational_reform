"""用户认证路由"""
import hashlib
from flask import Blueprint, render_template, jsonify, request, session, redirect, url_for
from models.models import db, User

auth_bp = Blueprint('auth', __name__)


def hash_password(password):
    """密码哈希"""
    return hashlib.sha256(password.encode('utf-8')).hexdigest()


@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    """登录页面"""
    if request.method == 'GET':
        return render_template('login.html')

    data = request.get_json() if request.is_json else request.form
    username = data.get('username', '')
    password = data.get('password', '')

    user = User.query.filter_by(username=username).first()
    if user and user.password_hash == hash_password(password):
        session['user_id'] = user.id
        session['username'] = user.username
        session['role'] = user.role
        user.last_login = db.func.now()
        db.session.commit()
        return jsonify({'code': 0, 'msg': '登录成功', 'data': user.to_dict()})
    else:
        return jsonify({'code': 1, 'msg': '用户名或密码错误'}), 401


@auth_bp.route('/logout')
def logout():
    """登出"""
    session.clear()
    return redirect(url_for('auth.login'))


@auth_bp.route('/api/users')
def api_users():
    """获取用户列表（仅管理员）"""
    if session.get('role') != 'admin':
        return jsonify({'code': 1, 'msg': '权限不足'}), 403
    users = User.query.all()
    return jsonify({'code': 0, 'data': [u.to_dict() for u in users]})


@auth_bp.route('/api/users/create', methods=['POST'])
def api_create_user():
    """创建用户（仅管理员）"""
    if session.get('role') != 'admin':
        return jsonify({'code': 1, 'msg': '权限不足'}), 403

    data = request.get_json()
    username = data.get('username', '')
    password = data.get('password', '')
    role = data.get('role', 'viewer')

    if User.query.filter_by(username=username).first():
        return jsonify({'code': 1, 'msg': '用户名已存在'})

    user = User(
        username=username,
        password_hash=hash_password(password),
        role=role,
        email=data.get('email', '')
    )
    db.session.add(user)
    db.session.commit()
    return jsonify({'code': 0, 'msg': '创建成功', 'data': user.to_dict()})


@auth_bp.route('/api/users/<int:user_id>/role', methods=['PUT'])
def api_update_role(user_id):
    """更新用户角色（仅管理员）"""
    if session.get('role') != 'admin':
        return jsonify({'code': 1, 'msg': '权限不足'}), 403

    user = User.query.get_or_404(user_id)
    data = request.get_json()
    user.role = data.get('role', user.role)
    db.session.commit()
    return jsonify({'code': 0, 'msg': '更新成功', 'data': user.to_dict()})


@auth_bp.route('/api/session')
def api_session_info():
    """获取当前会话信息"""
    if 'user_id' not in session:
        return jsonify({'code': 1, 'msg': '未登录'}), 401
    return jsonify({
        'code': 0,
        'data': {
            'user_id': session.get('user_id'),
            'username': session.get('username'),
            'role': session.get('role')
        }
    })
