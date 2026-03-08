"""
model.py
========
LightGBM 기반 분위수 회귀 모델을 정의합니다.

핵심:
- Quantile Regression을 통해 각 분위수 (q10, q50, q90)를 예측
- 3개의 독립적인 모델 학습 (각각 다른 alpha 파라미터)
- 모델 저장/로드 기능
- Feature importance 추출
"""

import lightgbm as lgb
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional
import joblib

from . import config

logger = config.setup_logger(__name__)

# ==================== 모델 정의 ====================

class QuantileRegressor:
    """
    LightGBM Quantile Regression 모델 래퍼

    각 분위수(alpha)에 대해 별도의 모델을 학습합니다.
    """

    def __init__(self, alpha: float = 0.5, random_seed: int = config.RANDOM_SEED):
        """
        Parameters
        ----------
        alpha : float
            분위수 (0.1, 0.5, 0.9 등)
        random_seed : int
            재현성을 위한 시드
        """
        self.alpha = alpha
        self.random_seed = random_seed
        self.model = None
        self.train_score = None
        self.val_score = None
        self.feature_importance = None
        self.feature_names = None

    def _get_params(self) -> Dict:
        """모델 파라미터를 반환합니다"""
        params = config.LGBM_PARAMS.copy()
        params['objective'] = 'regression'
        params['metric'] = 'l1'  # MAE
        params['alpha'] = self.alpha  # 분위수
        params['random_state'] = self.random_seed
        params['seed'] = self.random_seed

        return params

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        verbose: bool = True
    ) -> Tuple[float, float]:
        """
        모델을 학습합니다.

        Parameters
        ----------
        X_train : pd.DataFrame
            학습 피처
        y_train : pd.Series
            학습 타깃
        X_val : pd.DataFrame
            검증 피처
        y_val : pd.Series
            검증 타깃
        verbose : bool
            로깅 여부

        Returns
        -------
        Tuple[float, float]
            (train_score, val_score)
        """
        logger.info(f"\n모델 학습 시작 (alpha={self.alpha})...")

        params = self._get_params()

        # LightGBM 데이터셋 생성
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        # 모델 학습
        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=params['n_estimators'],
            valid_sets=[train_data, val_data],
            valid_names=['train', 'valid'],
            callbacks=[
                lgb.early_stopping(
                    stopping_rounds=config.EARLY_STOPPING_ROUNDS,
                    verbose=verbose
                ),
                lgb.log_evaluation(period=50) if verbose else lgb.log_evaluation(period=0)
            ]
        )

        # 피처명 저장
        self.feature_names = X_train.columns.tolist()

        # 성능 평가 (MAE 기준)
        y_train_pred = self.predict(X_train)
        y_val_pred = self.predict(X_val)

        train_mae = np.mean(np.abs(y_train - y_train_pred))
        val_mae = np.mean(np.abs(y_val - y_val_pred))

        self.train_score = train_mae
        self.val_score = val_mae

        logger.info(f"✓ 모델 학습 완료 (alpha={self.alpha})")
        logger.info(f"  Train MAE: {train_mae:.6f}")
        logger.info(f"  Val MAE:   {val_mae:.6f}")

        return train_mae, val_mae

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        예측을 수행합니다.

        Parameters
        ----------
        X : pd.DataFrame
            피처

        Returns
        -------
        np.ndarray
            예측값
        """
        if self.model is None:
            raise ValueError("모델이 학습되지 않았습니다")

        return self.model.predict(X)

    def get_feature_importance(self, importance_type: str = 'gain') -> pd.DataFrame:
        """
        피처 중요도를 반환합니다.

        Parameters
        ----------
        importance_type : str
            'gain' 또는 'split'

        Returns
        -------
        pd.DataFrame
            피처 중요도 (정렬됨)
        """
        if self.model is None:
            raise ValueError("모델이 학습되지 않았습니다")

        importance = self.model.feature_importance(importance_type=importance_type)
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)

        self.feature_importance = importance_df

        return importance_df

    def save(self, filepath: Path):
        """
        모델을 저장합니다.

        Parameters
        ----------
        filepath : Path
            저장 경로
        """
        if self.model is None:
            raise ValueError("저장할 모델이 없습니다")

        joblib.dump(self.model, filepath)
        logger.info(f"✓ 모델 저장: {filepath}")

    def load(self, filepath: Path):
        """
        모델을 로드합니다.

        Parameters
        ----------
        filepath : Path
            로드 경로
        """
        self.model = joblib.load(filepath)
        logger.info(f"✓ 모델 로드: {filepath}")

# ==================== 분위수 회귀 모델 세트 ====================

class QuantileRegressionModel:
    """
    q10, q50, q90 분위수 회귀 모델의 집합
    """

    def __init__(self, random_seed: int = config.RANDOM_SEED):
        """
        Parameters
        ----------
        random_seed : int
            재현성을 위한 시드
        """
        self.random_seed = random_seed
        self.models = {
            'q10': QuantileRegressor(alpha=0.1, random_seed=random_seed),
            'q50': QuantileRegressor(alpha=0.5, random_seed=random_seed),
            'q90': QuantileRegressor(alpha=0.9, random_seed=random_seed),
        }

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: Dict[str, pd.Series],
        X_val: pd.DataFrame,
        y_val: Dict[str, pd.Series],
        verbose: bool = True
    ) -> Dict[str, Tuple[float, float]]:
        """
        모든 분위수 모델을 학습합니다.

        Parameters
        ----------
        X_train : pd.DataFrame
            학습 피처
        y_train : Dict[str, pd.Series]
            학습 타깃 {'q10': Series, 'q50': Series, 'q90': Series}
        X_val : pd.DataFrame
            검증 피처
        y_val : Dict[str, pd.Series]
            검증 타깃
        verbose : bool
            로깅 여부

        Returns
        -------
        Dict[str, Tuple[float, float]]
            각 분위수별 (train_score, val_score)
        """
        results = {}

        for q in ['q10', 'q50', 'q90']:
            train_score, val_score = self.models[q].train(
                X_train, y_train[q], X_val, y_val[q], verbose=verbose
            )
            results[q] = (train_score, val_score)

        logger.info(f"\n✓ 모든 분위수 모델 학습 완료")

        return results

    def predict(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        모든 분위수에 대해 예측합니다.

        Parameters
        ----------
        X : pd.DataFrame
            피처

        Returns
        -------
        Dict[str, np.ndarray]
            {'q10': predictions, 'q50': predictions, 'q90': predictions}
        """
        predictions = {}

        for q in ['q10', 'q50', 'q90']:
            predictions[q] = self.models[q].predict(X)

        return predictions

    def predict_with_correction(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        예측을 수행하고, 분위수 순서가 맞지 않으면 보정합니다.

        q10 <= q50 <= q90 순서를 보장합니다.

        Parameters
        ----------
        X : pd.DataFrame
            피처

        Returns
        -------
        Dict[str, np.ndarray]
            {'q10': corrections_predictions, ...}
        """
        predictions = self.predict(X)

        if not config.QUANTILE_ORDER_CORRECTION:
            return predictions

        q10 = predictions['q10'].copy()
        q50 = predictions['q50'].copy()
        q90 = predictions['q90'].copy()

        # q10이 q50보다 크면 q10을 조정
        mask = q10 > q50
        if mask.any():
            q10[mask] = q50[mask] - np.abs(q50[mask]) * 0.01

        # q90이 q50보다 작으면 q90을 조정
        mask = q90 < q50
        if mask.any():
            q90[mask] = q50[mask] + np.abs(q50[mask]) * 0.01

        # q50이 q10 < q50 < q90 범위를 벗어나면 조정
        q50 = np.clip(q50, q10, q90)

        return {
            'q10': q10,
            'q50': q50,
            'q90': q90
        }

    def get_feature_importance(self) -> Dict[str, pd.DataFrame]:
        """
        모든 분위수 모델의 피처 중요도를 반환합니다.

        Returns
        -------
        Dict[str, pd.DataFrame]
            {'q10': importance_df, ...}
        """
        importance = {}

        for q in ['q10', 'q50', 'q90']:
            importance[q] = self.models[q].get_feature_importance()

        return importance

    def save(self, model_dir: Path):
        """
        모든 모델을 저장합니다.

        Parameters
        ----------
        model_dir : Path
            저장 디렉토리
        """
        model_dir.mkdir(parents=True, exist_ok=True)

        for q in ['q10', 'q50', 'q90']:
            filename = f"{config.TICKER}_model_{q}.pkl"
            filepath = model_dir / filename
            self.models[q].save(filepath)

        logger.info(f"✓ 모든 모델 저장 완료: {model_dir}")

    def load(self, model_dir: Path):
        """
        모든 모델을 로드합니다.

        Parameters
        ----------
        model_dir : Path
            로드 디렉토리
        """
        for q in ['q10', 'q50', 'q90']:
            filename = f"{config.TICKER}_model_{q}.pkl"
            filepath = model_dir / filename

            if filepath.exists():
                self.models[q].load(filepath)
            else:
                logger.warning(f"⚠ 모델 파일 없음: {filepath}")

        logger.info(f"✓ 모든 모델 로드 완료: {model_dir}")

# ==================== 테스트 함수 ====================

if __name__ == "__main__":
    logger.info("model 모듈 테스트 시작")

    # 더미 데이터 생성
    np.random.seed(config.RANDOM_SEED)
    X = pd.DataFrame(np.random.randn(1000, 10), columns=[f'feature_{i}' for i in range(10)])
    y_q10 = pd.Series(np.random.randn(1000) * 0.02)
    y_q50 = pd.Series(np.random.randn(1000) * 0.02)
    y_q90 = pd.Series(np.random.randn(1000) * 0.02)

    X_train, X_val = X[:800], X[800:]
    y_train_q10, y_val_q10 = y_q10[:800], y_q10[800:]
    y_train_q50, y_val_q50 = y_q50[:800], y_q50[800:]
    y_train_q90, y_val_q90 = y_q90[:800], y_q90[800:]

    # 모델 학습
    qr_model = QuantileRegressionModel()

    y_train = {'q10': y_train_q10, 'q50': y_train_q50, 'q90': y_train_q90}
    y_val = {'q10': y_val_q10, 'q50': y_val_q50, 'q90': y_val_q90}

    results = qr_model.train(X_train, y_train, X_val, y_val, verbose=False)

    logger.info("\n학습 결과:")
    for q, (train_score, val_score) in results.items():
        logger.info(f"  {q}: train={train_score:.6f}, val={val_score:.6f}")

    # 예측
    predictions = qr_model.predict(X_val)
    logger.info(f"\n예측 샘플 (처음 5개):")
    for q in ['q10', 'q50', 'q90']:
        logger.info(f"  {q}: {predictions[q][:5]}")

    # 피처 중요도
    importance = qr_model.get_feature_importance()
    logger.info(f"\n피처 중요도 (q50 상위 3개):")
    logger.info(importance['q50'].head(3))

    logger.info("\n✓ 모델 테스트 완료")
