"""
回归算法模块
支持：线性回归、岭回归、Lasso、GBDT
应用场景：股票价格预测、销售额预测
"""

import logging
import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, field

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error,
)


logger = logging.getLogger(__name__)


@dataclass
class RegressionResult:
    """回归结果数据结构"""
    model_name: str
    r2: float
    rmse: float
    mae: float
    mape: float
    cv_scores: List[float] = field(default_factory=list)
    cv_mean: float = 0.0
    cv_std: float = 0.0
    feature_importance: Optional[Dict[str, float]] = None
    coefficients: Optional[Dict[str, float]] = None
    y_pred: Optional[np.ndarray] = None
    model: Any = None

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "r2": round(self.r2, 4),
            "rmse": round(self.rmse, 4),
            "mae": round(self.mae, 4),
            "mape": round(self.mape, 4),
            "cv_mean": round(self.cv_mean, 4),
            "cv_std": round(self.cv_std, 4),
            "feature_importance": self.feature_importance,
        }


class RegressionModels:
    """回归算法集合"""

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
        """准备训练/测试数据"""
        if feature_cols is None:
            feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != target_col]

        X = df[feature_cols].values
        y = df[target_col].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )

        if scale:
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)

        logger.info(f"回归数据准备完成: 训练集 {X_train.shape}, 测试集 {X_test.shape}")
        return X_train, X_test, y_train, y_test, feature_cols

    def _evaluate(
        self,
        model,
        model_name: str,
        X_test: np.ndarray,
        y_test: np.ndarray,
        feature_names: List[str],
        cv: int = 5,
    ) -> RegressionResult:
        """评估回归模型"""
        y_pred = model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)

        mape = 0.0
        try:
            mape = mean_absolute_percentage_error(y_test, y_pred)
        except Exception:
            pass

        # 交叉验证
        cv_scores = []
        cv_mean, cv_std = 0.0, 0.0
        try:
            cv_scores = cross_val_score(model, X_test, y_test, cv=cv, scoring="r2")
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
        except Exception:
            pass

        # 特征重要性
        feature_importance = None
        coefficients = None
        if hasattr(model, "feature_importances_"):
            feature_importance = dict(zip(feature_names, model.feature_importances_.round(4)))
        if hasattr(model, "coef_"):
            coefficients = dict(zip(feature_names, model.coef_.round(4)))

        result = RegressionResult(
            model_name=model_name,
            r2=r2,
            rmse=rmse,
            mae=mae,
            mape=mape,
            cv_scores=cv_scores.tolist(),
            cv_mean=cv_mean,
            cv_std=cv_std,
            feature_importance=feature_importance,
            coefficients=coefficients,
            y_pred=y_pred,
            model=model,
        )

        logger.info(f"[{model_name}] R2={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}")
        return result

    def linear_regression(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        feature_names: List[str],
        cv: int = 5,
    ) -> RegressionResult:
        """线性回归"""
        model = LinearRegression()
        model.fit(X_train, y_train)
        self.models_["LinearRegression"] = model

        result = self._evaluate(model, "LinearRegression", X_test, y_test, feature_names, cv)
        self.results_["LinearRegression"] = result
        return result

    def ridge(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        feature_names: List[str],
        cv: int = 5,
        alpha: float = 1.0,
    ) -> RegressionResult:
        """岭回归"""
        model = Ridge(alpha=alpha, random_state=self.random_state)
        model.fit(X_train, y_train)
        self.models_["Ridge"] = model

        result = self._evaluate(model, "Ridge", X_test, y_test, feature_names, cv)
        self.results_["Ridge"] = result
        return result

    def lasso(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        feature_names: List[str],
        cv: int = 5,
        alpha: float = 1.0,
    ) -> RegressionResult:
        """Lasso 回归"""
        model = Lasso(alpha=alpha, random_state=self.random_state, max_iter=5000)
        model.fit(X_train, y_train)
        self.models_["Lasso"] = model

        result = self._evaluate(model, "Lasso", X_test, y_test, feature_names, cv)
        self.results_["Lasso"] = result
        return result

    def gbdt(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        feature_names: List[str],
        cv: int = 5,
        **kwargs,
    ) -> RegressionResult:
        """GBDT 梯度提升回归"""
        default_params = {
            "n_estimators": 100,
            "max_depth": 5,
            "learning_rate": 0.1,
            "random_state": self.random_state,
        }
        default_params.update(kwargs)

        model = GradientBoostingRegressor(**default_params)
        model.fit(X_train, y_train)
        self.models_["GBDT"] = model

        result = self._evaluate(model, "GBDT", X_test, y_test, feature_names, cv)
        self.results_["GBDT"] = result
        return result

    def compare_all(
        self,
        df: pd.DataFrame,
        target_col: str,
        feature_cols: Optional[List[str]] = None,
        test_size: float = 0.2,
        cv: int = 5,
    ) -> pd.DataFrame:
        """对比所有回归算法"""
        X_train, X_test, y_train, y_test, feature_names = self.prepare_data(
            df, target_col, feature_cols, test_size
        )

        self.linear_regression(X_train, X_test, y_train, y_test, feature_names, cv)
        self.ridge(X_train, X_test, y_train, y_test, feature_names, cv)
        self.lasso(X_train, X_test, y_train, y_test, feature_names, cv)
        self.gbdt(X_train, X_test, y_train, y_test, feature_names, cv)

        comparison = pd.DataFrame([r.to_dict() for r in self.results_.values()])
        comparison = comparison.sort_values("r2", ascending=False).reset_index(drop=True)

        print("\n" + "=" * 60)
        print("回归算法对比结果")
        print("=" * 60)
        print(comparison[["model_name", "r2", "rmse", "mae", "mape"]].to_string(index=False))
        print("=" * 60)

        return comparison

    def predict(self, model_name: str, X: np.ndarray, scale: bool = True) -> np.ndarray:
        """使用指定模型进行预测"""
        if model_name not in self.models_:
            raise ValueError(f"模型 '{model_name}' 不存在")
        if scale:
            X = self.scaler.transform(X)
        return self.models_[model_name].predict(X)
