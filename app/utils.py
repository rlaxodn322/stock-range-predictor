"""
utils.py
========
유틸리티 함수 모음
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime
from pathlib import Path
import json

from . import config

logger = config.setup_logger(__name__)

# ==================== 경로 관련 ====================

def ensure_dir(path: Path) -> Path:
    """
    디렉토리가 없으면 생성합니다.

    Parameters
    ----------
    path : Path
        생성할 경로

    Returns
    -------
    Path
        생성된 경로
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

# ==================== 데이터 관련 ====================

def save_dataframe_with_timestamp(
    df: pd.DataFrame,
    output_dir: Path,
    filename_prefix: str,
    format: str = 'csv'
) -> Path:
    """
    데이터프레임을 타임스탬프와 함께 저장합니다.

    Parameters
    ----------
    df : pd.DataFrame
        저장할 데이터프레임
    output_dir : Path
        출력 디렉토리
    filename_prefix : str
        파일명 접두사
    format : str
        저장 형식 ('csv', 'json', 'parquet')

    Returns
    -------
    Path
        저장된 파일 경로
    """
    output_dir = ensure_dir(output_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{filename_prefix}_{timestamp}.{format}"
    filepath = output_dir / filename

    if format == 'csv':
        df.to_csv(filepath, index=False, encoding='utf-8')
    elif format == 'json':
        df.to_json(filepath, orient='records', force_ascii=False)
    elif format == 'parquet':
        df.to_parquet(filepath, index=False)
    else:
        raise ValueError(f"지원하지 않는 형식: {format}")

    logger.info(f"✓ 저장: {filepath}")

    return filepath

# ==================== 통계 관련 ====================

def calculate_returns(prices: np.ndarray) -> np.ndarray:
    """
    가격에서 수익률을 계산합니다.

    Parameters
    ----------
    prices : np.ndarray
        가격 배열

    Returns
    -------
    np.ndarray
        수익률
    """
    return np.diff(prices) / prices[:-1]

def calculate_cumulative_return(returns: np.ndarray) -> float:
    """
    누적 수익률을 계산합니다.

    Parameters
    ----------
    returns : np.ndarray
        수익률 배열

    Returns
    -------
    float
        누적 수익률
    """
    return np.prod(1 + returns) - 1

def calculate_annualized_return(total_return: float, years: float) -> float:
    """
    연간화 수익률을 계산합니다.

    Parameters
    ----------
    total_return : float
        총 수익률
    years : float
        기간 (년)

    Returns
    -------
    float
        연간화 수익률
    """
    if years <= 0:
        return 0.0
    return (1 + total_return) ** (1 / years) - 1

def calculate_volatility(returns: np.ndarray, periods_per_year: int = 252) -> float:
    """
    변동성을 계산합니다.

    Parameters
    ----------
    returns : np.ndarray
        수익률 배열
    periods_per_year : int
        연간 기간 수

    Returns
    -------
    float
        연간화 변동성
    """
    return np.std(returns) * np.sqrt(periods_per_year)

def calculate_sharpe_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252
) -> float:
    """
    Sharpe Ratio를 계산합니다.

    Parameters
    ----------
    returns : np.ndarray
        수익률 배열
    risk_free_rate : float
        무위험 수익률
    periods_per_year : int
        연간 기간 수

    Returns
    -------
    float
        Sharpe Ratio
    """
    excess_returns = returns - (risk_free_rate / periods_per_year)
    volatility = calculate_volatility(returns, periods_per_year)

    if volatility == 0:
        return 0.0

    return np.mean(excess_returns) / volatility * np.sqrt(periods_per_year)

def calculate_max_drawdown(prices: np.ndarray) -> float:
    """
    최대 낙폭을 계산합니다.

    Parameters
    ----------
    prices : np.ndarray
        가격 배열

    Returns
    -------
    float
        최대 낙폭 (0 ~ -1)
    """
    cummax = np.maximum.accumulate(prices)
    drawdown = (prices - cummax) / cummax
    return np.min(drawdown)

# ==================== JSON 관련 ====================

def save_json(data: dict, filepath: Path):
    """
    데이터를 JSON으로 저장합니다.

    Parameters
    ----------
    data : dict
        저장할 데이터
    filepath : Path
        저장 경로
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    logger.info(f"✓ JSON 저장: {filepath}")

def load_json(filepath: Path) -> dict:
    """
    JSON 파일을 로드합니다.

    Parameters
    ----------
    filepath : Path
        로드 경로

    Returns
    -------
    dict
        로드된 데이터
    """
    filepath = Path(filepath)

    if not filepath.exists():
        logger.error(f"파일을 찾을 수 없습니다: {filepath}")
        return {}

    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    logger.info(f"✓ JSON 로드: {filepath}")

    return data

# ==================== 날짜 관련 ====================

def is_trading_day(date_str: str) -> bool:
    """
    주어진 날짜가 거래일인지 확인합니다.

    Parameters
    ----------
    date_str : str
        날짜 (YYYY-MM-DD)

    Returns
    -------
    bool
        거래일 여부
    """
    date_str = date_str.split(' ')[0]  # 시간 제거

    if date_str in config.KOREAN_HOLIDAYS:
        return False

    # 주말 확인
    date = pd.to_datetime(date_str)
    if date.dayofweek >= 5:  # 토요일(5), 일요일(6)
        return False

    return True

# ==================== 수치 관련 ====================

def round_to_nearest(value: float, nearest: float = 100) -> int:
    """
    값을 가장 가까운 단위로 반올림합니다.

    Parameters
    ----------
    value : float
        값
    nearest : float
        반올림 단위 (기본: 100원)

    Returns
    -------
    int
        반올림된 값
    """
    return int(np.round(value / nearest) * nearest)

# ==================== 로깅 관련 ====================

def log_dict(title: str, data: dict, level: int = logging.INFO):
    """
    딕셔너리를 로그로 출력합니다.

    Parameters
    ----------
    title : str
        제목
    data : dict
        딕셔너리
    level : int
        로그 레벨
    """
    logger.log(level, f"\n{title}")
    logger.log(level, "=" * 60)

    for key, value in data.items():
        if isinstance(value, float):
            logger.log(level, f"  {key}: {value:.6f}")
        elif isinstance(value, dict):
            logger.log(level, f"  {key}:")
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, float):
                    logger.log(level, f"    {sub_key}: {sub_value:.6f}")
                else:
                    logger.log(level, f"    {sub_key}: {sub_value}")
        else:
            logger.log(level, f"  {key}: {value}")

# ==================== 성능 평가 관련 ====================

def compare_strategies(
    strategy_returns: np.ndarray,
    benchmark_returns: np.ndarray,
    risk_free_rate: float = 0.02
) -> dict:
    """
    전략과 벤치마크를 비교합니다.

    Parameters
    ----------
    strategy_returns : np.ndarray
        전략 수익률
    benchmark_returns : np.ndarray
        벤치마크 수익률
    risk_free_rate : float
        무위험 수익률

    Returns
    -------
    dict
        비교 결과
    """
    strategy_total_return = calculate_cumulative_return(strategy_returns)
    benchmark_total_return = calculate_cumulative_return(benchmark_returns)

    strategy_sharpe = calculate_sharpe_ratio(strategy_returns, risk_free_rate)
    benchmark_sharpe = calculate_sharpe_ratio(benchmark_returns, risk_free_rate)

    strategy_mdd = calculate_max_drawdown(np.cumprod(1 + strategy_returns))
    benchmark_mdd = calculate_max_drawdown(np.cumprod(1 + benchmark_returns))

    outperformance = strategy_total_return - benchmark_total_return

    result = {
        'strategy_total_return': float(strategy_total_return),
        'benchmark_total_return': float(benchmark_total_return),
        'outperformance': float(outperformance),
        'strategy_sharpe': float(strategy_sharpe),
        'benchmark_sharpe': float(benchmark_sharpe),
        'sharpe_outperformance': float(strategy_sharpe - benchmark_sharpe),
        'strategy_mdd': float(strategy_mdd),
        'benchmark_mdd': float(benchmark_mdd),
        'mdd_improvement': float(benchmark_mdd - strategy_mdd),
    }

    return result

# ==================== 테스트 ====================

if __name__ == "__main__":
    logger.info("utils 모듈 테스트 시작")

    # 경로 테스트
    test_dir = ensure_dir(config.OUTPUTS_DIR / "test")
    logger.info(f"✓ 디렉토리 생성: {test_dir}")

    # 수치 함수 테스트
    prices = np.array([100, 102, 101, 105])
    returns = calculate_returns(prices)
    logger.info(f"✓ 수익률: {returns}")

    cumulative = calculate_cumulative_return(returns)
    logger.info(f"✓ 누적 수익률: {cumulative:.4f}")

    logger.info("\n✓ Utils 테스트 완료")
