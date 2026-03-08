"""
features.py
===========
피처 엔지니어링을 수행합니다.

주요 피처 그룹:
1. 가격 기반: 수익률, 갭, 범위, 캔들 형태
2. 이동평균/추세: MA, 기울기, 괴리율
3. 변동성: 표준편차, ATR
4. 모멘텀: RSI, MACD, Stochastic
5. 거래량: 거래량 변화율, 상대 거래량
6. 시장 확장: (코스피, 환율 등 - 향후 확장 가능)
"""

import pandas as pd
import numpy as np
from typing import List, Tuple
import logging

from . import config

logger = config.setup_logger(__name__)

# ==================== 가격 기반 피처 ====================

def add_price_features(df: pd.DataFrame, windows: List[int] = config.PRICE_WINDOWS) -> pd.DataFrame:
    """
    가격 기반 피처를 추가합니다.

    피처:
    - daily_return: 일간 수익률
    - return_nday: N일 수익률
    - gap: 전일 대비 시가 갭
    - high_low_ratio: (고가-저가)/종가 비율
    - close_position: (종가-저가)/(고가-저가) - 캔들 위치
    - body_ratio: 바디/(전체 범위) - 캔들 바디 크기
    - upper_tail_ratio: 위꼬리/전체범위
    - lower_tail_ratio: 아래꼬리/전체범위

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV 데이터
    windows : List[int]
        계산할 거래일 수

    Returns
    -------
    pd.DataFrame
        피처가 추가된 데이터
    """
    df = df.copy()

    # 1. 일간 수익률
    df['daily_return'] = df['Close'].pct_change()

    # 2. N일 수익률
    for w in windows:
        df[f'return_{w}d'] = df['Close'].pct_change(w)

    # 3. 전일 대비 시가 갭
    df['gap'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)

    # 4. 고가-저가 범위 비율
    df['high_low_ratio'] = (df['High'] - df['Low']) / df['Close']

    # 5. 캔들 위치값 (0: 바닥, 1: 천장)
    df['close_position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'] + 1e-8)
    df['close_position'] = df['close_position'].clip(0, 1)

    # 6. 캔들 바디 비율 (바디/전체범위)
    body = abs(df['Close'] - df['Open'])
    total_range = df['High'] - df['Low']
    df['body_ratio'] = body / (total_range + 1e-8)
    df['body_ratio'] = df['body_ratio'].clip(0, 1)

    # 7. 위꼬리 비율
    upper_tail = df['High'] - df[['Open', 'Close']].max(axis=1)
    df['upper_tail_ratio'] = upper_tail / (total_range + 1e-8)
    df['upper_tail_ratio'] = df['upper_tail_ratio'].clip(0, 1)

    # 8. 아래꼬리 비율
    lower_tail = df[['Open', 'Close']].min(axis=1) - df['Low']
    df['lower_tail_ratio'] = lower_tail / (total_range + 1e-8)
    df['lower_tail_ratio'] = df['lower_tail_ratio'].clip(0, 1)

    logger.info(f"✓ 가격 기반 피처 추가 완료")

    return df

# ==================== 이동평균/추세 피처 ====================

def add_ma_features(df: pd.DataFrame, windows: List[int] = config.MA_WINDOWS) -> pd.DataFrame:
    """
    이동평균 및 추세 피처를 추가합니다.

    피처:
    - ma_N: N일 이동평균
    - close_to_ma_N: 종가 / MA_N 비율
    - ma_slope_N: MA의 기울기 (최근 5일 변화율)
    - ma_divergence_fast_slow: 단기/중기 MA 괴리율

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV 데이터
    windows : List[int]
        MA 윈도우 크기

    Returns
    -------
    pd.DataFrame
        피처가 추가된 데이터
    """
    df = df.copy()

    for w in windows:
        # 이동평균
        df[f'ma_{w}'] = df['Close'].rolling(window=w, min_periods=1).mean()

        # 종가 / MA 비율
        df[f'close_to_ma_{w}'] = df['Close'] / (df[f'ma_{w}'] + 1e-8)

        # MA 기울기 (최근 5일 변화율)
        df[f'ma_slope_{w}'] = df[f'ma_{w}'].pct_change(5)

    # 단기/중기 MA 괴리율 (5일 vs 20일)
    if 5 in windows and 20 in windows:
        df['ma_divergence_5_20'] = (df['ma_5'] - df['ma_20']) / (df['ma_20'] + 1e-8)

    logger.info(f"✓ 이동평균/추세 피처 추가 완료")

    return df

# ==================== 변동성 피처 ====================

def add_volatility_features(df: pd.DataFrame, windows: List[int] = config.VOLATILITY_WINDOWS, atr_window: int = config.ATR_WINDOW) -> pd.DataFrame:
    """
    변동성 피처를 추가합니다.

    피처:
    - volatility_N: N일 rolling std
    - atr_N: Average True Range
    - recent_range_mean: 최근 N일 평균 진폭

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV 데이터
    windows : List[int]
        rolling window 크기
    atr_window : int
        ATR 윈도우

    Returns
    -------
    pd.DataFrame
        피처가 추가된 데이터
    """
    df = df.copy()

    # 1. Rolling volatility (일간 수익률의 표준편차)
    for w in windows:
        df[f'volatility_{w}'] = df['daily_return'].rolling(window=w, min_periods=1).std()

    # 2. True Range 계산 (ATR의 기반)
    df['tr'] = np.maximum(
        df['High'] - df['Low'],
        np.maximum(
            abs(df['High'] - df['Close'].shift(1)),
            abs(df['Low'] - df['Close'].shift(1))
        )
    )

    # 3. Average True Range
    df[f'atr_{atr_window}'] = df['tr'].rolling(window=atr_window, min_periods=1).mean()

    # 4. ATR 정규화 (close에 대한 % 비율)
    df[f'atr_pct_{atr_window}'] = df[f'atr_{atr_window}'] / df['Close']

    # 5. 최근 N일 평균 진폭
    for w in windows:
        df[f'range_mean_{w}'] = (df['High'] - df['Low']).rolling(window=w, min_periods=1).mean()
        df[f'range_mean_pct_{w}'] = df[f'range_mean_{w}'] / df['Close']

    # TR 컬럼 제거 (필요 없음)
    df = df.drop('tr', axis=1)

    logger.info(f"✓ 변동성 피처 추가 완료")

    return df

# ==================== 모멘텀 피처 ====================

def add_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """
    RSI (Relative Strength Index)를 계산합니다.

    Parameters
    ----------
    series : pd.Series
        가격 시리즈
    window : int
        윈도우 크기

    Returns
    -------
    pd.Series
        RSI 값 (0~100)
    """
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window, min_periods=1).mean()

    rs = gain / (loss + 1e-8)
    rsi = 100 - (100 / (1 + rs))

    return rsi

def add_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    MACD (Moving Average Convergence Divergence)를 계산합니다.

    Parameters
    ----------
    series : pd.Series
        가격 시리즈
    fast : int
        빠른 EMA 윈도우
    slow : int
        느린 EMA 윈도우
    signal : int
        시그널 라인 윈도우

    Returns
    -------
    Tuple[pd.Series, pd.Series, pd.Series]
        (MACD, Signal, Histogram)
    """
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()

    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram

def add_stochastic(df: pd.DataFrame, window: int = 14, smooth_k: int = 3, smooth_d: int = 3) -> Tuple[pd.Series, pd.Series]:
    """
    Stochastic Oscillator를 계산합니다.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV 데이터
    window : int
        윈도우 크기
    smooth_k : int
        K 스무딩 윈도우
    smooth_d : int
        D 스무딩 윈도우

    Returns
    -------
    Tuple[pd.Series, pd.Series]
        (%K, %D)
    """
    low_min = df['Low'].rolling(window=window, min_periods=1).min()
    high_max = df['High'].rolling(window=window, min_periods=1).max()

    k_raw = 100 * (df['Close'] - low_min) / (high_max - low_min + 1e-8)
    k = k_raw.rolling(window=smooth_k, min_periods=1).mean()
    d = k.rolling(window=smooth_d, min_periods=1).mean()

    return k, d

def add_momentum_features(df: pd.DataFrame, rsi_window: int = config.RSI_WINDOW,
                         macd_fast: int = config.MACD_FAST, macd_slow: int = config.MACD_SLOW,
                         macd_signal: int = config.MACD_SIGNAL) -> pd.DataFrame:
    """
    모멘텀 피처를 추가합니다.

    피처:
    - rsi_14: RSI
    - macd: MACD 라인
    - macd_signal: MACD 시그널 라인
    - macd_histogram: MACD 히스토그램
    - stochastic_k: Stochastic %K
    - stochastic_d: Stochastic %D

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV 데이터

    Returns
    -------
    pd.DataFrame
        피처가 추가된 데이터
    """
    df = df.copy()

    # 1. RSI
    df[f'rsi_{rsi_window}'] = add_rsi(df['Close'], window=rsi_window)

    # 2. MACD
    macd, macd_sig, macd_hist = add_macd(df['Close'], fast=macd_fast, slow=macd_slow, signal=macd_signal)
    df['macd'] = macd
    df['macd_signal'] = macd_sig
    df['macd_histogram'] = macd_hist

    # 3. Stochastic
    stoch_k, stoch_d = add_stochastic(df, window=14)
    df['stochastic_k'] = stoch_k
    df['stochastic_d'] = stoch_d

    logger.info(f"✓ 모멘텀 피처 추가 완료")

    return df

# ==================== 거래량 피처 ====================

def add_volume_features(df: pd.DataFrame, windows: List[int] = [5, 10, 20]) -> pd.DataFrame:
    """
    거래량 피처를 추가합니다.

    피처:
    - volume_change: 거래량 변화율
    - volume_ma_N: N일 거래량 이동평균
    - volume_to_ma_N: 거래량 / 거래량 MA 비율 (상대 거래량)
    - volume_trend: 거래량 트렌드

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV 데이터
    windows : List[int]
        MA 윈도우

    Returns
    -------
    pd.DataFrame
        피처가 추가된 데이터
    """
    df = df.copy()

    # 1. 거래량 변화율
    df['volume_change'] = df['Volume'].pct_change()

    # 2. 거래량 이동평균 및 상대 거래량
    for w in windows:
        df[f'volume_ma_{w}'] = df['Volume'].rolling(window=w, min_periods=1).mean()
        df[f'volume_to_ma_{w}'] = df['Volume'] / (df[f'volume_ma_{w}'] + 1e-8)

    logger.info(f"✓ 거래량 피처 추가 완료")

    return df

# ==================== 결측치 처리 ====================

def handle_feature_nan(df: pd.DataFrame, min_rows: int = config.MIN_DATA_ROWS_FOR_FEATURES) -> pd.DataFrame:
    """
    피처 계산으로 인한 결측치를 처리합니다.

    이동평균, 변동성 등의 계산 후 초반 행들에 NaN이 생기므로,
    이를 제거합니다.

    Parameters
    ----------
    df : pd.DataFrame
        피처가 포함된 데이터
    min_rows : int
        최소 필요 행 수

    Returns
    -------
    pd.DataFrame
        결측치가 제거된 데이터
    """
    initial_rows = len(df)

    # NaN 제거
    df = df.dropna()
    df = df.reset_index(drop=True)

    rows_removed = initial_rows - len(df)
    logger.info(f"✓ 피처 결측치 처리 완료: {rows_removed}행 제거, {len(df)}행 남음")

    if len(df) < min_rows:
        logger.warning(f"⚠ 남은 데이터가 최소 필요량보다 적습니다: {len(df)} < {min_rows}")

    return df

# ==================== 메인 피처 엔지니어링 함수 ====================

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    모든 피처를 엔지니어링합니다.

    **중요 원칙**:
    - 모든 피처는 "오늘까지 알 수 있는 정보"만 사용
    - 미래 데이터 누수 절대 금지
    - 내일 값을 유추하지 않음

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV 데이터 (반드시 Date 컬럼 포함)

    Returns
    -------
    pd.DataFrame
        모든 피처가 추가된 데이터
    """
    logger.info("피처 엔지니어링 시작...")

    if 'daily_return' not in df.columns:
        df = add_price_features(df)
    else:
        df = df.copy()

    df = add_ma_features(df)
    df = add_volatility_features(df)
    df = add_momentum_features(df)
    df = add_volume_features(df)

    # 결측치 처리
    df = handle_feature_nan(df)

    logger.info(f"✓ 피처 엔지니어링 완료: {len(df)} rows, {len(df.columns)} columns")

    return df

# ==================== 피처 선택 함수 ====================

def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """
    모델 학습에 사용할 피처 컬럼을 반환합니다.

    제외 컬럼:
    - Date, Open, High, Low, Close, Volume (이미 사용됨)
    - 타깃 변수들
    - 기타 메타데이터

    Parameters
    ----------
    df : pd.DataFrame
        피처가 포함된 데이터

    Returns
    -------
    List[str]
        피처 컬럼 리스트
    """
    exclude_cols = [
        'Date', 'Open', 'High', 'Low', 'Close', 'Volume',
        'next_return_q10', 'next_return_q50', 'next_return_q90',
        'DayOfWeek', 'Month', 'Quarter', 'Year', 'DayOfMonth'
    ]

    feature_cols = [col for col in df.columns if col not in exclude_cols]

    return feature_cols

# ==================== 테스트 함수 ====================

if __name__ == "__main__":
    from .data_loader import load_and_prepare_data

    logger.info("features 모듈 테스트 시작")

    df = load_and_prepare_data(start_date="2023-01-01", end_date="2024-12-31")

    if not df.empty:
        df = engineer_features(df)

        logger.info("\n피처 샘플:")
        feature_cols = get_feature_columns(df)
        logger.info(df[['Date'] + feature_cols[:5]].head(10))

        logger.info(f"\n전체 피처: {len(feature_cols)}")
        logger.info(feature_cols)
    else:
        logger.error("테스트 실패")
