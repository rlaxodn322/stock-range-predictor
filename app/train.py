"""
train.py
========
모델 학습 파이프라인을 구현합니다.

절차:
1. 데이터 수집
2. 피처 엔지니어링
3. 타깃 생성
4. 데이터 분리
5. 모델 학습
6. 결과 저장
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from datetime import datetime

from . import config
from .data_loader import load_and_prepare_data, split_train_val_test
from .features import engineer_features, get_feature_columns
from .targets import create_target_variables, validate_targets, print_target_statistics
from .model import QuantileRegressionModel

logger = config.setup_logger(__name__)

# ==================== 학습 파이프라인 ====================

def run_training_pipeline(
    ticker: str = config.TICKER,
    start_date: str = config.DATA_START_DATE,
    end_date: str = config.DATA_END_DATE,
    save_models: bool = True,
    save_importance: bool = True
) -> tuple:
    """
    전체 학습 파이프라인을 실행합니다.

    Parameters
    ----------
    ticker : str
        종목 코드
    start_date : str
        데이터 시작 날짜
    end_date : str
        데이터 종료 날짜
    save_models : bool
        모델 저장 여부
    save_importance : bool
        피처 중요도 저장 여부

    Returns
    -------
    tuple
        (model, train_results, val_results, feature_importance, df_test)
    """
    logger.info("=" * 80)
    logger.info("모델 학습 파이프라인 시작")
    logger.info("=" * 80)

    # ============ 1단계: 데이터 수집 ============
    logger.info("\n[1/5] 데이터 수집 중...")
    df = load_and_prepare_data(ticker=ticker, start_date=start_date, end_date=end_date)

    if df.empty:
        logger.error("데이터 수집 실패")
        return None, None, None, None, None

    logger.info(f"✓ 데이터 수집 완료: {len(df)} rows")

    # ============ 2단계: 피처 엔지니어링 ============
    logger.info("\n[2/5] 피처 엔지니어링 중...")
    df = engineer_features(df)

    if df.empty:
        logger.error("피처 엔지니어링 실패")
        return None, None, None, None, None

    # ============ 3단계: 타깃 생성 ============
    logger.info("\n[3/5] 타깃 생성 중...")
    df, target_q10, target_q50, target_q90 = create_target_variables(df)

    # 타깃 검증
    if not validate_targets(target_q10, target_q50, target_q90):
        logger.error("타깃 검증 실패")
        return None, None, None, None, None

    print_target_statistics(target_q10, target_q50, target_q90)

    # ============ 4단계: 데이터 분리 ============
    logger.info("\n[4/5] 데이터 분리 중...")

    df_train, df_val, df_test = split_train_val_test(df)

    # 각 세트에서 타깃 추출
    target_q10_train = df_train['next_return_q10']
    target_q50_train = df_train['next_return_q50']
    target_q90_train = df_train['next_return_q90']

    target_q10_val = df_val['next_return_q10']
    target_q50_val = df_val['next_return_q50']
    target_q90_val = df_val['next_return_q90']

    target_q10_test = df_test['next_return_q10']
    target_q50_test = df_test['next_return_q50']
    target_q90_test = df_test['next_return_q90']

    # 피처 추출
    feature_cols = get_feature_columns(df_train)

    X_train = df_train[feature_cols]
    X_val = df_val[feature_cols]
    X_test = df_test[feature_cols]

    logger.info(f"✓ 데이터 분리 완료")
    logger.info(f"  피처: {len(feature_cols)}")
    logger.info(f"  샘플: {len(X_train)} (train) + {len(X_val)} (val) + {len(X_test)} (test)")

    # ============ 5단계: 모델 학습 ============
    logger.info("\n[5/5] 모델 학습 중...")

    y_train = {
        'q10': target_q10_train,
        'q50': target_q50_train,
        'q90': target_q90_train
    }

    y_val = {
        'q10': target_q10_val,
        'q50': target_q50_val,
        'q90': target_q90_val
    }

    # 분위수 회귀 모델 생성 및 학습
    qr_model = QuantileRegressionModel(random_seed=config.RANDOM_SEED)
    train_results = qr_model.train(X_train, y_train, X_val, y_val, verbose=True)

    logger.info(f"\n✓ 모델 학습 완료")

    # 테스트 성능 평가
    logger.info("\n테스트 세트 성능:")
    predictions_test = qr_model.predict(X_test)

    test_mae_q10 = np.mean(np.abs(target_q10_test - predictions_test['q10']))
    test_mae_q50 = np.mean(np.abs(target_q50_test - predictions_test['q50']))
    test_mae_q90 = np.mean(np.abs(target_q90_test - predictions_test['q90']))

    logger.info(f"  q10 MAE: {test_mae_q10:.6f}")
    logger.info(f"  q50 MAE: {test_mae_q50:.6f}")
    logger.info(f"  q90 MAE: {test_mae_q90:.6f}")

    val_results = {
        'q10': test_mae_q10,
        'q50': test_mae_q50,
        'q90': test_mae_q90
    }

    # ============ 모델 저장 ============
    if save_models:
        logger.info("\n모델 저장 중...")
        qr_model.save(config.MODELS_DIR)

    # ============ 피처 중요도 저장 ============
    if save_importance:
        logger.info("\n피처 중요도 저장 중...")
        feature_importance = qr_model.get_feature_importance()

        for q in ['q10', 'q50', 'q90']:
            output_path = config.OUTPUTS_DIR / f"{config.OUTPUT_FEATURE_IMPORTANCE_PREFIX}_{q}.csv"
            feature_importance[q].to_csv(output_path, index=False)
            logger.info(f"✓ 저장: {output_path}")

    # ============ 최종 로그 ============
    logger.info("\n" + "=" * 80)
    logger.info("모델 학습 파이프라인 완료")
    logger.info("=" * 80)

    return qr_model, train_results, val_results, feature_importance, df_test

# ==================== 편의 함수 ====================

def train_model(
    ticker: str = config.TICKER,
    start_date: str = config.DATA_START_DATE,
    end_date: str = config.DATA_END_DATE
) -> QuantileRegressionModel:
    """
    모델을 학습하는 편의 함수

    Parameters
    ----------
    ticker : str
        종목 코드
    start_date : str
        시작 날짜
    end_date : str
        종료 날짜

    Returns
    -------
    QuantileRegressionModel
        학습된 모델
    """
    model, _, _, _, _ = run_training_pipeline(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        save_models=True,
        save_importance=True
    )

    return model

# ==================== 테스트 함수 ====================

if __name__ == "__main__":
    logger.info("train 모듈 테스트 시작")

    # 최근 2년 데이터로 학습
    model, train_results, val_results, importance, df_test = run_training_pipeline(
        start_date="2022-01-01",
        end_date="2024-12-31"
    )

    if model is not None:
        logger.info("\n✓ 학습 완료")
    else:
        logger.error("✗ 학습 실패")
