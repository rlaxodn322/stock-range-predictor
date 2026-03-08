"""
predict.py
==========
다음 거래일 종가 범위를 예측합니다.

절차:
1. 최신 데이터 수집 및 피처 계산
2. 학습된 모델로 수익률 분위수 예측
3. 예측 수익률을 종가 범위로 변환
4. 신호 생성
5. 결과 저장
"""

import pandas as pd
import numpy as np
import logging
import json
from pathlib import Path
from datetime import datetime, timedelta

from . import config
from .data_loader import load_and_prepare_data
from .features import engineer_features, get_feature_columns
from .model import QuantileRegressionModel

logger = config.setup_logger(__name__)

# ==================== 예측 함수 ====================

def load_trained_models(model_dir: Path = config.MODELS_DIR) -> QuantileRegressionModel:
    """
    저장된 모델을 로드합니다.

    Parameters
    ----------
    model_dir : Path
        모델 디렉토리

    Returns
    -------
    QuantileRegressionModel
        로드된 모델
    """
    logger.info(f"모델 로드 중... ({model_dir})")

    qr_model = QuantileRegressionModel()

    try:
        qr_model.load(model_dir)
        logger.info(f"✓ 모델 로드 완료")
        return qr_model
    except Exception as e:
        logger.error(f"✗ 모델 로드 실패: {e}")
        return None

def get_latest_data(
    ticker: str = config.TICKER,
    lookback_days: int = 365
) -> pd.DataFrame:
    """
    최신 데이터를 수집합니다.

    Parameters
    ----------
    ticker : str
        종목 코드
    lookback_days : int
        과거 데이터 기간

    Returns
    -------
    pd.DataFrame
        최신 데이터 (피처 포함)
    """
    logger.info("최신 데이터 수집 중...")

    # 현재 날짜 기준 lookback 기간 설정
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

    # 데이터 수집
    df = load_and_prepare_data(ticker=ticker, start_date=start_date, end_date=end_date)

    if df.empty:
        logger.error("데이터 수집 실패")
        return pd.DataFrame()

    # 피처 엔지니어링
    df = engineer_features(df)

    logger.info(f"✓ 데이터 준비 완료: {len(df)} rows")
    logger.info(f"  최신 거래일: {df['Date'].max()}")

    return df

def predict_next_range(
    df: pd.DataFrame,
    model: QuantileRegressionModel
) -> dict:
    """
    다음 거래일 종가 범위를 예측합니다.

    Parameters
    ----------
    df : pd.DataFrame
        최신 데이터 (피처 포함)
    model : QuantileRegressionModel
        학습된 모델

    Returns
    -------
    dict
        예측 결과
    """
    if df.empty or model is None:
        logger.error("데이터 또는 모델이 없습니다")
        return None

    logger.info("\n예측 수행 중...")

    # 최신 데이터 (마지막 행)
    latest_idx = len(df) - 1
    latest_data = df.iloc[latest_idx]
    today_close = latest_data['Close']

    logger.info(f"  기준일: {latest_data['Date']}")
    logger.info(f"  금일 종가: {today_close:,.0f}원")

    # 피처 추출
    feature_cols = get_feature_columns(df)
    X_latest = df[[c for c in feature_cols if c in df.columns]].iloc[[-1]]

    # 수익률 분위수 예측 (보정 포함)
    predictions = model.predict_with_correction(X_latest)

    q10_return = predictions['q10'][0]
    q50_return = predictions['q50'][0]
    q90_return = predictions['q90'][0]

    logger.info(f"\n예측 수익률:")
    logger.info(f"  q10: {q10_return:.6f} ({q10_return*100:.2f}%)")
    logger.info(f"  q50: {q50_return:.6f} ({q50_return*100:.2f}%)")
    logger.info(f"  q90: {q90_return:.6f} ({q90_return*100:.2f}%)")

    # 수익률을 종가 범위로 변환
    low_close = today_close * (1 + q10_return)
    mid_close = today_close * (1 + q50_return)
    high_close = today_close * (1 + q90_return)

    # 반올림 (원 단위)
    low_close = int(np.round(low_close))
    mid_close = int(np.round(mid_close))
    high_close = int(np.round(high_close))

    logger.info(f"\n예측 종가 범위:")
    logger.info(f"  하단(low):  {low_close:,}원")
    logger.info(f"  중심(mid):  {mid_close:,}원")
    logger.info(f"  상단(high): {high_close:,}원")

    # 범위 폭 계산
    range_width = high_close - low_close
    range_width_pct = (high_close - low_close) / mid_close

    logger.info(f"\n범위 폭:")
    logger.info(f"  절대값: {range_width:,}원")
    logger.info(f"  상대값: {range_width_pct*100:.2f}%")

    # 신호 생성
    signal = generate_signal(q10_return, q50_return, q90_return)
    logger.info(f"\n신호: {signal}")

    # 결과 반환
    result = {
        'ticker': config.TICKER,
        'prediction_for_next_trading_day': (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d"),
        'today_date': latest_data['Date'].strftime("%Y-%m-%d"),
        'today_close': int(today_close),
        'predicted_return_quantiles': {
            'q10': float(q10_return),
            'q50': float(q50_return),
            'q90': float(q90_return),
        },
        'predicted_close_range': {
            'low': low_close,
            'mid': mid_close,
            'high': high_close,
        },
        'range_width': range_width,
        'range_width_pct': float(range_width_pct),
        'signal': signal,
        'timestamp': datetime.now().isoformat()
    }

    return result

def generate_signal(q10: float, q50: float, q90: float) -> str:
    """
    예측 결과 기반으로 거래 신호를 생성합니다.

    신호 규칙:
    - BULLISH: q50 > 0.8% 그리고 q10 > -0.5%
    - BEARISH: q50 < -0.8% 그리고 q90 < 0.3%
    - NEUTRAL: 기타

    Parameters
    ----------
    q10, q50, q90 : float
        예측 수익률 분위수

    Returns
    -------
    str
        신호 ('BULLISH', 'BEARISH', 'NEUTRAL')
    """
    if q50 > config.SIGNAL_BULLISH_Q50 and q10 > config.SIGNAL_BULLISH_Q10:
        return 'BULLISH'
    elif q50 < -config.SIGNAL_BEARISH_Q50 and q90 < config.SIGNAL_BEARISH_Q90:
        return 'BEARISH'
    else:
        return 'NEUTRAL'

# ==================== 결과 저장 ====================

def save_prediction(prediction: dict, output_dir: Path = config.OUTPUTS_DIR):
    """
    예측 결과를 저장합니다.

    Parameters
    ----------
    prediction : dict
        예측 결과
    output_dir : Path
        출력 디렉토리
    """
    if prediction is None:
        logger.error("저장할 예측이 없습니다")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON 저장
    json_path = output_dir / config.OUTPUT_LATEST_PREDICTION_JSON
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(prediction, f, ensure_ascii=False, indent=2)

    logger.info(f"✓ 예측 저장: {json_path}")

    # CSV 저장 (추가 기록용)
    csv_path = output_dir / "predictions_history.csv"

    prediction_row = {
        'timestamp': prediction['timestamp'],
        'prediction_date': prediction['prediction_for_next_trading_day'],
        'today_close': prediction['today_close'],
        'predicted_low': prediction['predicted_close_range']['low'],
        'predicted_mid': prediction['predicted_close_range']['mid'],
        'predicted_high': prediction['predicted_close_range']['high'],
        'range_width': prediction['range_width'],
        'q10': prediction['predicted_return_quantiles']['q10'],
        'q50': prediction['predicted_return_quantiles']['q50'],
        'q90': prediction['predicted_return_quantiles']['q90'],
        'signal': prediction['signal'],
    }

    df_row = pd.DataFrame([prediction_row])

    if csv_path.exists():
        df_history = pd.read_csv(csv_path)
        df_history = pd.concat([df_history, df_row], ignore_index=True)
    else:
        df_history = df_row

    df_history.to_csv(csv_path, index=False, encoding='utf-8')
    logger.info(f"✓ 예측 기록 저장: {csv_path}")

# ==================== 메인 예측 함수 ====================

def run_prediction(
    ticker: str = config.TICKER,
    model_dir: Path = config.MODELS_DIR,
    output_dir: Path = config.OUTPUTS_DIR,
    lookback_days: int = 365,
    save_result: bool = True
) -> dict:
    """
    전체 예측 파이프라인을 실행합니다.

    Parameters
    ----------
    ticker : str
        종목 코드
    model_dir : Path
        모델 디렉토리
    output_dir : Path
        출력 디렉토리
    lookback_days : int
        과거 데이터 기간
    save_result : bool
        결과 저장 여부

    Returns
    -------
    dict
        예측 결과
    """
    logger.info("=" * 80)
    logger.info("예측 파이프라인 시작")
    logger.info("=" * 80)

    # 1. 모델 로드
    model = load_trained_models(model_dir)
    if model is None:
        logger.error("모델 로드 실패")
        return None

    # 2. 최신 데이터 수집
    df = get_latest_data(ticker, lookback_days)
    if df.empty:
        logger.error("데이터 수집 실패")
        return None

    # 3. 예측
    prediction = predict_next_range(df, model)
    if prediction is None:
        logger.error("예측 실패")
        return None

    # 4. 결과 저장
    if save_result:
        save_prediction(prediction, output_dir)

    logger.info("\n" + "=" * 80)
    logger.info("예측 파이프라인 완료")
    logger.info("=" * 80)

    return prediction

# ==================== 테스트 함수 ====================

if __name__ == "__main__":
    logger.info("predict 모듈 테스트 시작")

    prediction = run_prediction()

    if prediction is not None:
        logger.info("\n예측 결과:")
        logger.info(json.dumps(prediction, ensure_ascii=False, indent=2))
    else:
        logger.error("✗ 예측 실패")
