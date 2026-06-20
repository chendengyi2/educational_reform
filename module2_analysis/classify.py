"""
分类算法模块
支持：逻辑回归、随机森林、XGBoost、SVM
应用场景：信用风险评估、客户流失预测
"""

import logging
import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, field

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix,
)


logger = logging.getLogger(__name__)


@dataclass
class ClassificationResult:
    """分类结果数据结构"""
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    auc: float = 0.0
    cv_scores: List[float] = field(default_factory=list)
    cv_mean: float = 0.0
    cv_std: float = 0.0
    confusion_matrix: Optional[np.ndarray] = None
    feature_importance: Optional[Dict[str, float]] = None
    report: str = ""
    model: Any = None
    predict_proba: Optional[np.ndarray] = None

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "accuracy": round(self.accuracy, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "auc": round(self.auc, 4),
            "cv_mean": round(self.cv_mean, 4),
            "cv_std": round(self.cv_std, 4),
            "feature_importance": self.feature_importance,
        }


class ClassificationModels:
    """分类算法集合"""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.models_ = {}
        self.results_ = {}

    def prepare_data(
        self,
        df: pd.DataFrame,
        target_col: str,
        feature_cols: Optional[List[str]] = None,
        test_size: float = 0.2,
        scale: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """
        准备训练/测试数据

        Args:
            df: 输入DataFrame
            target_col: 目标列
            feature_cols: 特征列
            test_size: 测试集比例
            scale: 是否标准化

        Returns:
            X_train, X_test, y_train, y_test, feature_names
        """
        if feature_cols is None:
            feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != target_col]

        X = df[feature_cols].values
        y = df[target_col].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )

        if scale:
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)

        logger.info(f"数据准备完成: 训练集 {X_train.shape}, 测试集 {X_test.shape}")
        return X_train, X_test, y_train, y_test, feature_cols

    def _evaluate(
        self,
        model,
        model_name: str,
        X_test: np.ndarray,
        y_test: np.ndarray,
        feature_names: List[str],
        cv: int = 5,
    ) -> ClassificationResult:
        """评估分类模型"""
        y_pred = model.predict(X_test)

        # 基础指标
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        # AUC
        auc = 0.0
        predict_proba = None
        if hasattr(model, "predict_proba"):
            try:
                predict_proba = model.predict_proba(X_test)
                if predict_proba.shape[1] == 2:
                    auc = roc_auc_score(y_test, predict_proba[:, 1])
                else:
                    auc = roc_auc_score(y_test, predict_proba, multi_class="ovr")
            except Exception:
                pass

        # 交叉验证
        cv_scores = []
        cv_mean, cv_std = 0.0, 0.0
        try:
            cv_scores = cross_val_score(model, X_test, y_test, cv=cv, scoring="accuracy")
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
        except Exception:
            pass

        # 特征重要性
        feature_importance = None
        if hasattr(model, "feature_importances_"):
            feature_importance = dict(zip(feature_names, model.feature_importances_.round(4)))
        elif hasattr(model, "coef_"):
            coef = model.coef_[0] if model.coef_.ndim > 1 else model.coef_
            feature_importance = dict(zip(feature_names, np.abs(coef).round(4)))

        # 混淆矩阵与分类报告
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, zero_division=0)

        result = ClassificationResult(
            model_name=model_name,
            accuracy=acc,
            precision=prec,
            recall=rec,
            f1=f1,
            auc=auc,
            cv_scores=cv_scores.tolist(),
            cv_mean=cv_mean,
            cv_std=cv_std,
            confusion_matrix=cm,
            feature_importance=feature_importance,
            report=report,
            model=model,
            predict_proba=predict_proba,
        )

        logger.info(f"[{model_name}] Acc={acc:.4f}, F1={f1:.4f}, AUC={auc:.4f}")
        return result

    def logistic_regression(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        feature_names: List[str],
        cv: int = 5,
        **kwargs,
    ) -> ClassificationResult:
        """逻辑回归分类"""
        default_params = {"max_iter": 1000, "random_state": self.random_state}
        default_params.update(kwargs)

        model = LogisticRegression(**default_params)
        model.fit(X_train, y_train)
        self.models_["LogisticRegression"] = model

        result = self._evaluate(model, "LogisticRegression", X_test, y_test, feature_names, cv)
        self.results_["LogisticRegression"] = result
        return result

    def random_forest(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        feature_names: List[str],
        cv: int = 5,
        **kwargs,
    ) -> ClassificationResult:
        """随机森林分类"""
        default_params = {
            "n_estimators": 100,
            "max_depth": 10,
            "random_state": self.random_state,
            "n_jobs": -1,
        }
        default_params.update(kwargs)

        model = RandomForestClassifier(**default_params)
        model.fit(X_train, y_train)
        self.models_["RandomForest"] = model

        result = self._evaluate(model, "RandomForest", X_test, y_test, feature_names, cv)
        self.results_["RandomForest"] = result
        return result

    def xgboost(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        feature_names: List[str],
        cv: int = 5,
        **kwargs,
    ) -> Optional[ClassificationResult]:
        """XGBoost 分类"""
        try:
            from xgboost import XGBClassifier
        except ImportError:
            logger.warning("XGBoost 未安装，跳过。安装: pip install xgboost")
            return None

        default_params = {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "random_state": self.random_state,
            "use_label_encoder": False,
            "eval_metric": "logloss",
        }
        default_params.update(kwargs)

        model = XGBClassifier(**default_params)
        model.fit(X_train, y_train)
        self.models_["XGBoost"] = model

        result = self._evaluate(model, "XGBoost", X_test, y_test, feature_names, cv)
        self.results_["XGBoost"] = result
        return result

    def svm(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        feature_names: List[str],
        cv: int = 5,
        **kwargs,
    ) -> ClassificationResult:
        """SVM 分类"""
        default_params = {
            "kernel": "rbf",
            "probability": True,
            "random_state": self.random_state,
        }
        default_params.update(kwargs)

        model = SVC(**default_params)
        model.fit(X_train, y_train)
        self.models_["SVM"] = model

        result = self._evaluate(model, "SVM", X_test, y_test, feature_names, cv)
        self.results_["SVM"] = result
        return result

    def compare_all(
        self,
        df: pd.DataFrame,
        target_col: str,
        feature_cols: Optional[List[str]] = None,
        test_size: float = 0.2,
        cv: int = 5,
    ) -> pd.DataFrame:
        """
        对比所有分类算法

        Args:
            df: 输入DataFrame
            target_col: 目标列
            feature_cols: 特征列
            test_size: 测试集比例
            cv: 交叉验证折数

        Returns:
            对比结果DataFrame
        """
        X_train, X_test, y_train, y_test, feature_names = self.prepare_data(
            df, target_col, feature_cols, test_size
        )

        # 运行所有模型
        self.logistic_regression(X_train, X_test, y_train, y_test, feature_names, cv)
        self.random_forest(X_train, X_test, y_train, y_test, feature_names, cv)
        self.xgboost(X_train, X_test, y_train, y_test, feature_names, cv)
        self.svm(X_train, X_test, y_train, y_test, feature_names, cv)

        # 生成对比表
        comparison = pd.DataFrame([r.to_dict() for r in self.results_.values()])
        comparison = comparison.sort_values("f1", ascending=False).reset_index(drop=True)

        print("\n" + "=" * 70)
        print("分类算法对比结果")
        print("=" * 70)
        print(comparison[["model_name", "accuracy", "precision", "recall", "f1", "auc"]].to_string(index=False))
        print("=" * 70)

        return comparison

    def predict(self, model_name: str, X: np.ndarray, scale: bool = True) -> np.ndarray:
        """使用指定模型进行预测"""
        if model_name not in self.models_:
            raise ValueError(f"模型 '{model_name}' 不存在，可用: {list(self.models_.keys())}")

        if scale:
            X = self.scaler.transform(X)

        return self.models_[model_name].predict(X)
