"""
main.py
=======
프로젝트의 메인 엔트리포인트

이 모듈은 데이터 로드, 모델 학습, 예측, 평가, 백테스트를 조율합니다.
"""

import logging
from pathlib import Path
import pandas as pd
import numpy as np

from . import config
from .data_loader import load_and_prepare_data, split_train_val_test
from .features import engineer_features, get_feature_columns
from .targets import create_target_variables
from .train import run_training_pipeline
from .predict import run_prediction, get_latest_data
from .evaluation import evaluate_quantile_predictions, plot_prediction_band, plot_actual_vs_predicted, save_evaluation_results
from .backtest import run_backtest
from .model import QuantileRegressionModel

logger = config.setup_logger(__name__)

# ==================== 메인 함수 ====================

def main_train(
    ticker: str = config.TICKER,
    start_date: str = config.DATA_START_DATE,
    end_date: str = config.DATA_END_DATE
):
    """
    모델 학습을 실행합니다.

    Parameters
    ----------
    ticker : str
        종목 코드
    start_date : str
        시작 날짜
    end_date : str
        종료 날짜
    """
    logger.info("\n" + "=" * 80)
    logger.info("MAIN: 모델 학습 모드")
    logger.info("=" * 80)

    model, train_results, val_results, importance, df_test = run_training_pipeline(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        save_models=True,
        save_importance=True
    )

    if model is None:
        logger.error("모델 학습 실패")
        return

    logger.info("\n✓ 모델 학습 완료")

def main_predict(ticker: str = config.TICKER):
    """
    예측을 실행합니다.

    Parameters
    ----------
    ticker : str
        종목 코드
    """
    logger.info("\n" + "=" * 80)
    logger.info("MAIN: 예측 모드")
    logger.info("=" * 80)

    prediction = run_prediction(ticker=ticker)

    if prediction is None:
        logger.error("예측 실패")
        return

    logger.info("\n✓ 예측 완료")

def main_evaluate(
    ticker: str = config.TICKER,
    start_date: str = config.DATA_START_DATE,
    end_date: str = config.DATA_END_DATE
):
    """
    모델 성능을 평가합니다.

    Parameters
    ----------
    ticker : str
        종목 코드
    start_date : str
        시작 날짜
    end_date : str
        종료 날짜
    """
    logger.info("\n" + "=" * 80)
    logger.info("MAIN: 평가 모드")
    logger.info("=" * 80)

    # 1. 데이터 준비
    logger.info("\n[1/4] 데이터 준비 중...")
    df = load_and_prepare_data(ticker=ticker, start_date=start_date, end_date=end_date)

    if df.empty:
        logger.error("데이터 로드 실패")
        return

    df = engineer_features(df)
    df, target_q10, target_q50, target_q90 = create_target_variables(df)

    df_train, df_val, df_test = split_train_val_test(df)

    # 2. 모델 로드 및 예측
    logger.info("\n[2/4] 모델 로드 및 테스트 예측 중...")

    model = QuantileRegressionModel()
    model.load(config.MODELS_DIR)

    feature_cols = get_feature_columns(df_test)
    X_test = df_test[feature_cols]

    predictions = model.predict_with_correction(X_test)
    y_true = df_test['next_return_q50'].values

    # 3. 평가
    logger.info("\n[3/4] 성능 평가 중...")

    results = evaluate_quantile_predictions(
        y_true,
        predictions['q10'],
        predictions['q50'],
        predictions['q90']
    )

    # 4. 시각화 및 저장
    logger.info("\n[4/4] 시각화 및 저장 중...")

    # 예측 범위 차트
    output_path = config.OUTPUTS_DIR / config.OUTPUT_PREDICTION_BAND_PNG
    plot_prediction_band(
        df_test['Date'],
        y_true,
        predictions['q10'],
        predictions['q50'],
        predictions['q90'],
        output_path
    )

    # 실제 vs 예측 차트
    output_path = config.OUTPUTS_DIR / config.OUTPUT_ACTUAL_VS_PREDICTED_PNG
    plot_actual_vs_predicted(
        df_test['Date'],
        y_true,
        predictions['q50'],
        output_path
    )

    # 평가 결과 저장
    save_evaluation_results(results, config.OUTPUTS_DIR)

    logger.info("\n✓ 평가 완료")

def main_backtest(
    ticker: str = config.TICKER,
    start_date: str = config.DATA_START_DATE,
    end_date: str = config.DATA_END_DATE
):
    """
    백테스트를 실행합니다.

    Parameters
    ----------
    ticker : str
        종목 코드
    start_date : str
        시작 날짜
    end_date : str
        종료 날짜
    """
    logger.info("\n" + "=" * 80)
    logger.info("MAIN: 백테스트 모드")
    logger.info("=" * 80)

    # 1. 데이터 준비
    logger.info("\n[1/3] 데이터 준비 중...")
    df = load_and_prepare_data(ticker=ticker, start_date=start_date, end_date=end_date)

    if df.empty:
        logger.error("데이터 로드 실패")
        return

    df = engineer_features(df)
    df, target_q10, target_q50, target_q90 = create_target_variables(df)

    # 2. 모델 로드 및 예측
    logger.info("\n[2/3] 모델 로드 및 예측 중...")

    model = QuantileRegressionModel()
    model.load(config.MODELS_DIR)

    feature_cols = get_feature_columns(df)
    X = df[feature_cols]

    predictions = model.predict_with_correction(X)

    # 3. 백테스트
    logger.info("\n[3/3] 백테스트 실행 중...")

    performance, df_result = run_backtest(
        df,
        predictions,
        initial_capital=1000000,
        output_dir=config.OUTPUTS_DIR
    )

    logger.info("\n✓ 백테스트 완료")

def main_full_pipeline(
    ticker: str = config.TICKER,
    start_date: str = config.DATA_START_DATE,
    end_date: str = config.DATA_END_DATE
):
    """
    전체 파이프라인을 실행합니다 (학습 + 평가 + 백테스트).

    Parameters
    ----------
    ticker : str
        종목 코드
    start_date : str
        시작 날짜
    end_date : str
        종료 날짜
    """
    logger.info("\n" + "=" * 80)
    logger.info("MAIN: 전체 파이프라인 모드")
    logger.info("=" * 80)

    # 1. 학습
    main_train(ticker, start_date, end_date)

    # 2. 평가
    main_evaluate(ticker, start_date, end_date)

    # 3. 백테스트
    main_backtest(ticker, start_date, end_date)

    # 4. 예측
    main_predict(ticker)

    logger.info("\n" + "=" * 80)
    logger.info("✓ 전체 파이프라인 완료")
    logger.info("=" * 80)

# ==================== 편의 함수 ====================

def run_mode(
    mode: str = "train",
    ticker: str = config.TICKER,
    start_date: str = config.DATA_START_DATE,
    end_date: str = config.DATA_END_DATE
):
    """
    지정된 모드를 실행합니다.

    Parameters
    ----------
    mode : str
        실행 모드 ('train', 'predict', 'evaluate', 'backtest', 'full')
    ticker : str
        종목 코드
    start_date : str
        시작 날짜
    end_date : str
        종료 날짜
    """
    mode = mode.lower()

    if mode == "train":
        main_train(ticker, start_date, end_date)
    elif mode == "predict":
        main_predict(ticker)
    elif mode == "evaluate":
        main_evaluate(ticker, start_date, end_date)
    elif mode == "backtest":
        main_backtest(ticker, start_date, end_date)
    elif mode == "full":
        main_full_pipeline(ticker, start_date, end_date)
    else:
        logger.error(f"알 수 없는 모드: {mode}")
        logger.info("사용 가능한 모드: train, predict, evaluate, backtest, full")

if __name__ == "__main__":
    run_mode("train")
