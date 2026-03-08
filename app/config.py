"""
config.py
=========
프로젝트의 모든 설정 및 상수를 정의합니다.
"""

from pathlib import Path
from datetime import datetime, timedelta
import logging

# ==================== 경로 설정 ====================
PROJECT_ROOT = Path(__file__).parent.parent
APP_DIR = PROJECT_ROOT / "app"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# 디렉토리 생성
MODELS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)

# ==================== 로깅 설정 ====================
LOG_LEVEL = logging.INFO
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

def setup_logger(name: str) -> logging.Logger:
    """로거 설정 함수"""
    logger = logging.getLogger(name)
    logger.setLevel(LOG_LEVEL)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(LOG_FORMAT)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger

# ==================== 데이터 설정 ====================
# 삼성전자 종목 코드
TICKER = "005930"
TICKER_NAME = "삼성전자"

# 데이터 수집 기간 (기본값)
# 최소 10년 이상의 데이터를 수집하도록 설정
DATA_START_DATE = "2014-01-01"
DATA_END_DATE = datetime.now().strftime("%Y-%m-%d")

# 데이터 소스
DATA_SOURCE = "pykrx"  # "pykrx" 또는 "yfinance"
FALLBACK_SOURCE = "yfinance"

# yfinance fallback 시 사용할 티커
YFINANCE_TICKER = "005930.KS"

# ==================== 데이터 전처리 설정 ====================
# 결측치 처리
DROPNA_THRESHOLD = 0.95  # 95% 이상 데이터가 있어야 함
MAX_MISSING_RATIO = 0.05  # 5% 이상 결측이면 경고

# 이상치 감지
OUTLIER_METHOD = "iqr"  # "iqr" 또는 "zscore"
OUTLIER_THRESHOLD = 3.0  # zscore 기준

# ==================== 피처 엔지니어링 설정 ====================
# 롤링 윈도우 크기
PRICE_WINDOWS = [2, 3, 5, 10, 20, 60]
MA_WINDOWS = [5, 10, 20, 60]
VOLATILITY_WINDOWS = [5, 10, 20]
RSI_WINDOW = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
ATR_WINDOW = 14

# 피처 생성 시 최소 필요 데이터 행
MIN_DATA_ROWS_FOR_FEATURES = max(PRICE_WINDOWS + MA_WINDOWS + VOLATILITY_WINDOWS + [RSI_WINDOW, MACD_SLOW, ATR_WINDOW])

# ==================== 타깃 설정 ====================
# 예측 대상: 다음 거래일 수익률
# target = (next_close / today_close) - 1
TARGET_QUANTILES = [0.1, 0.5, 0.9]  # 10분위, 중앙값, 90분위
TARGET_NAME = "next_return"

# ==================== 데이터 분리 설정 ====================
# 시계열 기반 분리 (절대 랜덤 분할 금지)
TRAIN_RATIO = 0.70      # 70%
VAL_RATIO = 0.15       # 15%
TEST_RATIO = 0.15      # 15%

# 검증: TRAIN_RATIO + VAL_RATIO + TEST_RATIO = 1.0
assert abs((TRAIN_RATIO + VAL_RATIO + TEST_RATIO) - 1.0) < 1e-6, "분할 비율의 합이 1이 아닙니다"

# ==================== 모델 설정 ====================
# LightGBM 하이퍼파라미터
LGBM_PARAMS = {
    "objective": "regression",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "n_estimators": 500,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_samples": 20,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
    "verbose": -1,
}

# Random seed (재현성)
RANDOM_SEED = 42

# 조기 중단 설정
EARLY_STOPPING_ROUNDS = 50
EARLY_STOPPING_METRIC = "l1"  # MAE 기반

# ==================== 예측 설정 ====================
# 예측 분위수에 대한 임계값 검증
# q10 <= q50 <= q90 순서가 유지되어야 함
QUANTILE_ORDER_CHECK = True

# 예측 값이 이상하면 보정할지 여부
QUANTILE_ORDER_CORRECTION = True

# 종가 범위 환산 방식
# close_price = today_close * (1 + return)
CLOSE_ROUNDING = "round"  # "round" 또는 "ceil" 또는 "floor"

# ==================== 평가 설정 ====================
# Coverage 평가
COVERAGE_THRESHOLD = 0.90  # 커버리지가 이 이상이어야 함 (경고 기준)

# 구간 폭 평가
MAX_RANGE_WIDTH_PCT = 0.05  # 5% 이상 폭이면 경고

# ==================== 백테스트 설정 ====================
# 거래 비용
TRANSACTION_COST_PCT = 0.001  # 0.1% (왕복)

# 진입/청산 규칙
# 다음 거래일 시가에 진입하고, 같은 날 종가에 청산
BACKTEST_ENTRY_TYPE = "next_open"  # "next_open" 또는 "today_close"
BACKTEST_EXIT_TYPE = "same_day_close"  # "same_day_close" 또는 "next_open"

# 거래 신호 기준
BACKTEST_ENTRY_THRESHOLD_Q50 = 0.005  # q50이 0.5% 이상 상승 예상
BACKTEST_ENTRY_THRESHOLD_Q10 = -0.005  # q10이 -0.5% 이상 (너무 하락 예상하지 않음)

# Signal 분류
SIGNAL_BULLISH_Q50 = 0.008  # 0.8% 이상 상승 예상
SIGNAL_BULLISH_Q10 = -0.005  # q10이 -0.5% 이상
SIGNAL_BEARISH_Q50 = -0.008  # 0.8% 이상 하락 예상
SIGNAL_BEARISH_Q90 = 0.003  # q90이 0.3% 이상 (상승 여지 부족)

# ==================== 출력 설정 ====================
# 저장할 결과 파일명
OUTPUT_PREDICTIONS_CSV = "predictions.csv"
OUTPUT_EVALUATION_JSON = "evaluation_metrics.json"
OUTPUT_EVALUATION_CSV = "evaluation_metrics.csv"
OUTPUT_FEATURE_IMPORTANCE_PREFIX = "feature_importance"
OUTPUT_EQUITY_CURVE_CSV = "equity_curve.csv"
OUTPUT_EQUITY_CURVE_PNG = "equity_curve.png"
OUTPUT_PREDICTION_BAND_PNG = "prediction_band_plot.png"
OUTPUT_ACTUAL_VS_PREDICTED_PNG = "actual_vs_predicted.png"
OUTPUT_LATEST_PREDICTION_JSON = "latest_prediction.json"
OUTPUT_BACKTEST_SUMMARY_JSON = "backtest_summary.json"

# ==================== 모델 파일명 ====================
MODEL_Q10_NAME = f"{TICKER}_model_q10.pkl"
MODEL_Q50_NAME = f"{TICKER}_model_q50.pkl"
MODEL_Q90_NAME = f"{TICKER}_model_q90.pkl"

# ==================== 시각화 설정 ====================
FIGURE_DPI = 100
FIGURE_SIZE_WIDE = (14, 6)
FIGURE_SIZE_SQUARE = (10, 8)
PLOT_STYLE = "seaborn-v0_8-darkgrid"

# ==================== 날짜 설정 ====================
# 한국 주식 시장 휴장일 (간단한 예시)
# 실제로는 한국거래소 공식 휴장일을 사용해야 함
KOREAN_HOLIDAYS = [
    "2024-01-01",  # 신정
    "2024-02-09", "2024-02-10", "2024-02-11", "2024-02-12",  # 설날
    "2024-03-01",  # 삼일절
    "2024-04-10",  # 총선
    "2024-05-05", "2024-05-06",  # 어린이날, 대체
    "2024-05-15",  # 부처님오신날
    "2024-06-06",  # 현충일
    "2024-08-15",  # 광복절
    "2024-09-16", "2024-09-17", "2024-09-18",  # 추석
    "2024-10-03",  # 개천절
    "2024-10-09",  # 한글날
    "2024-12-25",  # 크리스마스
    "2025-01-01",
    "2025-01-29", "2025-01-30", "2025-01-31", "2025-02-01",
    "2025-03-01",
    "2025-05-05", "2025-05-06",
    "2025-05-15",
    "2025-06-06",
    "2025-08-15",
    "2025-09-17", "2025-09-18", "2025-09-19",
    "2025-10-03",
    "2025-10-09",
    "2025-12-25",
    "2026-01-01",
    "2026-02-17", "2026-02-18", "2026-02-19", "2026-02-20",
    "2026-03-01",
]

# ==================== 디버그 설정 ====================
DEBUG_MODE = False
VERBOSE = True

# 데이터 수집 시 캐시 사용 여부
USE_DATA_CACHE = False
CACHE_DIR = PROJECT_ROOT / ".cache"
CACHE_DIR.mkdir(exist_ok=True)

print(f"Configuration loaded: {PROJECT_ROOT}")
