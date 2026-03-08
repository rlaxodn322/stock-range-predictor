"""
data_loader.py
==============
한국 주식 데이터를 수집하고 전처리합니다.

주요 기능:
- pykrx를 통한 데이터 수집 (주 데이터 소스)
- yfinance를 통한 fallback 수집
- 데이터 검증 및 전처리
- 결측치, 중복, 이상치 처리
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Tuple, Optional
import logging
import warnings

from . import config

logger = config.setup_logger(__name__)

# ==================== 데이터 수집 함수 ====================

def fetch_data_pykrx(
    ticker: str,
    start_date: str,
    end_date: str
) -> pd.DataFrame:
    """
    pykrx를 통해 한국 주식 데이터를 수집합니다.

    Parameters
    ----------
    ticker : str
        종목 코드 (예: "005930")
    start_date : str
        시작 날짜 (YYYY-MM-DD)
    end_date : str
        종료 날짜 (YYYY-MM-DD)

    Returns
    -------
    pd.DataFrame
        컬럼: ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    """
    try:
        from pykrx import stock

        logger.info(f"pykrx로 {ticker} 데이터 수집 중... ({start_date} ~ {end_date})")

        # pykrx는 대신증권 API를 사용하여 한국 주식 데이터를 제공
        df = stock.get_market_ohlcv(start_date, end_date, ticker)

        if df.empty:
            logger.warning(f"pykrx에서 {ticker} 데이터를 찾을 수 없습니다.")
            return pd.DataFrame()

        # 컬럼명 정규화
        df = df.reset_index()
        # pykrx는 reset_index 후 7개 컬럼 반환 (추가 거래대금 컬럼)
        df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Amount']
        df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]

        # Date를 datetime으로 변환
        df['Date'] = pd.to_datetime(df['Date'])

        logger.info(f"pykrx 데이터 수집 완료: {len(df)} rows")

        return df

    except Exception as e:
        logger.error(f"pykrx 데이터 수집 오류: {e}")
        return pd.DataFrame()

def fetch_data_yfinance(
    ticker: str,
    start_date: str,
    end_date: str
) -> pd.DataFrame:
    """
    yfinance를 통해 한국 주식 데이터를 수집합니다. (Fallback)

    Parameters
    ----------
    ticker : str
        yfinance 티커 (예: "005930.KS")
    start_date : str
        시작 날짜 (YYYY-MM-DD)
    end_date : str
        종료 날짜 (YYYY-MM-DD)

    Returns
    -------
    pd.DataFrame
        컬럼: ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    """
    try:
        import yfinance as yf

        logger.info(f"yfinance로 {ticker} 데이터 수집 중... ({start_date} ~ {end_date})")

        df = yf.download(ticker, start=start_date, end=end_date, progress=False)

        if df.empty:
            logger.warning(f"yfinance에서 {ticker} 데이터를 찾을 수 없습니다.")
            return pd.DataFrame()

        # 컬럼명 정규화
        # yfinance는 MultiIndex 컬럼을 반환하므로 처리 필요
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]

        df = df.reset_index()
        df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]

        # Date를 datetime으로 변환
        df['Date'] = pd.to_datetime(df['Date'])

        logger.info(f"yfinance 데이터 수집 완료: {len(df)} rows")

        return df

    except Exception as e:
        logger.error(f"yfinance 데이터 수집 오류: {e}")
        return pd.DataFrame()

def load_data(
    ticker: str = config.TICKER,
    start_date: str = config.DATA_START_DATE,
    end_date: str = config.DATA_END_DATE,
    primary_source: str = config.DATA_SOURCE,
    fallback_source: str = config.FALLBACK_SOURCE,
) -> pd.DataFrame:
    """
    주식 데이터를 수집합니다 (primary_source를 시도하고, 실패 시 fallback 사용)

    Parameters
    ----------
    ticker : str
        종목 코드
    start_date : str
        시작 날짜
    end_date : str
        종료 날짜
    primary_source : str
        주 데이터 소스 ("pykrx" 또는 "yfinance")
    fallback_source : str
        백업 데이터 소스

    Returns
    -------
    pd.DataFrame
        수집된 데이터
    """
    df = pd.DataFrame()

    # Primary source로 시도
    if primary_source == "pykrx":
        df = fetch_data_pykrx(ticker, start_date, end_date)
        if not df.empty:
            logger.info(f"✓ pykrx 데이터 수집 성공")
            return df
        else:
            logger.warning(f"✗ pykrx 데이터 수집 실패, fallback 시도 중...")

    # Fallback source
    if fallback_source == "yfinance":
        yfinance_ticker = f"{ticker}.KS"
        df = fetch_data_yfinance(yfinance_ticker, start_date, end_date)
        if not df.empty:
            logger.info(f"✓ yfinance fallback 데이터 수집 성공")
            return df
        else:
            logger.error(f"✗ 모든 데이터 소스 실패")
            return pd.DataFrame()

    if df.empty:
        logger.error("데이터 수집 실패")
        return pd.DataFrame()

    return df

# ==================== 데이터 전처리 함수 ====================

def validate_data(df: pd.DataFrame) -> bool:
    """
    데이터의 기본 유효성을 검사합니다.

    Parameters
    ----------
    df : pd.DataFrame
        검사할 데이터프레임

    Returns
    -------
    bool
        유효성 여부
    """
    if df.empty:
        logger.error("데이터프레임이 비어있습니다")
        return False

    required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    for col in required_columns:
        if col not in df.columns:
            logger.error(f"필수 컬럼 '{col}'이 없습니다")
            return False

    if len(df) < 100:
        logger.error(f"데이터가 너무 적습니다: {len(df)} rows (최소 100 필요)")
        return False

    logger.info(f"✓ 데이터 검증 통과: {len(df)} rows")
    return True

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    데이터를 정제합니다.

    처리 항목:
    - 날짜 정렬
    - 중복 제거
    - 결측치 확인
    - 이상치 감지 (선택적 경고)

    Parameters
    ----------
    df : pd.DataFrame
        정제할 데이터

    Returns
    -------
    pd.DataFrame
        정제된 데이터
    """
    df = df.copy()

    logger.info("데이터 정제 시작...")

    # 1. 날짜로 정렬
    df = df.sort_values('Date').reset_index(drop=True)
    logger.info(f"✓ 날짜 기준 정렬 완료")

    # 2. 중복 제거
    duplicates = df.duplicated(subset=['Date']).sum()
    if duplicates > 0:
        logger.warning(f"⚠ 중복된 날짜 {duplicates}개 발견 및 제거")
        df = df.drop_duplicates(subset=['Date'], keep='last')
        df = df.reset_index(drop=True)

    # 3. 결측치 확인
    missing_count = df.isnull().sum().sum()
    if missing_count > 0:
        logger.warning(f"⚠ 결측치 {missing_count}개 발견")
        logger.warning(f"  컬럼별 결측치:\n{df.isnull().sum()}")

        # 결측치 제거
        df = df.dropna()
        df = df.reset_index(drop=True)
        logger.info(f"✓ 결측치 제거 완료: {len(df)} rows 남음")

    # 4. 데이터 타입 확인 및 변환
    df['Date'] = pd.to_datetime(df['Date'])
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # 수정된 결측치 제거
    df = df.dropna()
    df = df.reset_index(drop=True)

    # 5. 가격 유효성 확인 (High >= Low, Close 유효 등)
    invalid_high_low = (df['High'] < df['Low']).sum()
    if invalid_high_low > 0:
        logger.warning(f"⚠ High < Low인 행 {invalid_high_low}개 발견")
        df = df[df['High'] >= df['Low']]
        df = df.reset_index(drop=True)

    # 6. 이상치 감지 (가격이 극단적으로 변동한 경우 경고)
    df['daily_return'] = df['Close'].pct_change()
    extreme_returns = (abs(df['daily_return']) > 0.1).sum()
    if extreme_returns > 0:
        logger.warning(f"⚠ 10% 이상 변동한 날짜 {extreme_returns}개 발견 (정상범위 내)")

    # 수익률 컬럼 제거 (아직은 필요 없음)
    df = df.drop('daily_return', axis=1)

    logger.info(f"✓ 데이터 정제 완료: 최종 {len(df)} rows")
    logger.info(f"  기간: {df['Date'].min()} ~ {df['Date'].max()}")

    return df

def add_trading_day_info(df: pd.DataFrame) -> pd.DataFrame:
    """
    거래일 정보를 추가합니다.

    Parameters
    ----------
    df : pd.DataFrame
        데이터

    Returns
    -------
    pd.DataFrame
        거래일 정보가 추가된 데이터
    """
    df = df.copy()

    df['DayOfWeek'] = df['Date'].dt.day_name()
    df['Month'] = df['Date'].dt.month
    df['Quarter'] = df['Date'].dt.quarter
    df['Year'] = df['Date'].dt.year
    df['DayOfMonth'] = df['Date'].dt.day

    return df

# ==================== 데이터 분리 함수 ====================

def split_train_val_test(
    df: pd.DataFrame,
    train_ratio: float = config.TRAIN_RATIO,
    val_ratio: float = config.VAL_RATIO,
    test_ratio: float = config.TEST_RATIO
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    시계열 데이터를 시간 순서대로 train / validation / test로 분리합니다.

    **중요**: 절대 랜덤 셔플을 하지 않습니다. 시계열 데이터의 무결성을 보장합니다.

    Parameters
    ----------
    df : pd.DataFrame
        데이터
    train_ratio : float
        학습 데이터 비율
    val_ratio : float
        검증 데이터 비율
    test_ratio : float
        테스트 데이터 비율

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        (train_df, val_df, test_df)
    """
    n = len(df)
    train_size = int(n * train_ratio)
    val_size = int(n * val_ratio)

    train_df = df[:train_size].reset_index(drop=True)
    val_df = df[train_size:train_size + val_size].reset_index(drop=True)
    test_df = df[train_size + val_size:].reset_index(drop=True)

    logger.info(f"데이터 분리 완료:")
    logger.info(f"  Train: {len(train_df)} rows ({train_ratio*100:.1f}%) [{train_df['Date'].min()} ~ {train_df['Date'].max()}]")
    logger.info(f"  Val:   {len(val_df)} rows ({val_ratio*100:.1f}%) [{val_df['Date'].min()} ~ {val_df['Date'].max()}]")
    logger.info(f"  Test:  {len(test_df)} rows ({test_ratio*100:.1f}%) [{test_df['Date'].min()} ~ {test_df['Date'].max()}]")

    return train_df, val_df, test_df

# ==================== 메인 데이터 로드 함수 ====================

def load_and_prepare_data(
    ticker: str = config.TICKER,
    start_date: str = config.DATA_START_DATE,
    end_date: str = config.DATA_END_DATE,
    add_features: bool = False  # features 함수와의 중복 방지
) -> pd.DataFrame:
    """
    데이터를 수집, 검증, 정제합니다.

    Parameters
    ----------
    ticker : str
        종목 코드
    start_date : str
        시작 날짜
    end_date : str
        종료 날짜
    add_features : bool
        거래일 정보 추가 여부

    Returns
    -------
    pd.DataFrame
        준비된 데이터
    """
    # 1. 데이터 수집
    df = load_data(ticker, start_date, end_date)

    if df.empty:
        logger.error("데이터 수집 실패")
        return pd.DataFrame()

    # 2. 데이터 검증
    if not validate_data(df):
        logger.error("데이터 검증 실패")
        return pd.DataFrame()

    # 3. 데이터 정제
    df = clean_data(df)

    # 4. 거래일 정보 추가 (옵션)
    if add_features:
        df = add_trading_day_info(df)

    return df

# ==================== 테스트 함수 ====================

if __name__ == "__main__":
    # 테스트
    logger.info("data_loader 모듈 테스트 시작")

    df = load_and_prepare_data(
        ticker=config.TICKER,
        start_date="2023-01-01",
        end_date="2024-12-31",
        add_features=True
    )

    if not df.empty:
        logger.info("\n데이터 샘플:")
        logger.info(df.head(10))
        logger.info(f"\n데이터 통계:\n{df[['Open', 'High', 'Low', 'Close', 'Volume']].describe()}")
    else:
        logger.error("테스트 실패")
