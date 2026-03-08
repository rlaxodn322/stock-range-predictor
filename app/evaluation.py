"""
evaluation.py
=============
예측 성능을 평가합니다.

평가 지표:
1. MAE / RMSE / MAPE (중앙값 예측 기준)
2. Coverage Rate (범위 내 적중률)
3. Range Width (구간 폭)
4. Directional Accuracy (방향성 적중률)
"""

import pandas as pd
import numpy as np
import logging
import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from . import config

logger = config.setup_logger(__name__)

# ==================== 평가 지표 함수 ====================

def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error"""
    return np.mean(np.abs(y_true - y_pred))

def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error"""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error"""
    # 0 값 처리
    mask = np.abs(y_true) > 1e-8
    if mask.sum() == 0:
        return 0.0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))

def calculate_smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric Mean Absolute Percentage Error"""
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    mask = denominator > 1e-8
    if mask.sum() == 0:
        return 0.0
    return 2.0 * np.mean(np.abs((y_true[mask] - y_pred[mask]) / denominator[mask]))

def calculate_coverage_rate(
    y_true: np.ndarray,
    y_low: np.ndarray,
    y_high: np.ndarray
) -> float:
    """
    실제 값이 [low, high] 범위에 들어온 비율

    Parameters
    ----------
    y_true : np.ndarray
        실제 값
    y_low : np.ndarray
        예측 하단
    y_high : np.ndarray
        예측 상단

    Returns
    -------
    float
        coverage rate (0~1)
    """
    in_range = (y_true >= y_low) & (y_true <= y_high)
    return np.mean(in_range)

def calculate_range_width(y_low: np.ndarray, y_high: np.ndarray) -> float:
    """
    예측 범위의 평균 폭

    Parameters
    ----------
    y_low : np.ndarray
        예측 하단
    y_high : np.ndarray
        예측 상단

    Returns
    -------
    float
        평균 범위 폭 (절대값)
    """
    return np.mean(y_high - y_low)

def calculate_range_width_pct(
    y_low: np.ndarray,
    y_high: np.ndarray,
    y_mid: np.ndarray
) -> float:
    """
    예측 범위의 평균 폭 (상대값)

    Parameters
    ----------
    y_low, y_high, y_mid : np.ndarray
        예측값

    Returns
    -------
    float
        평균 범위 폭 (%)
    """
    return np.mean((y_high - y_low) / (np.abs(y_mid) + 1e-8))

def calculate_directional_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    baseline: float = 0.0
) -> float:
    """
    방향성 적중률

    Parameters
    ----------
    y_true : np.ndarray
        실제 값
    y_pred : np.ndarray
        예측값 (중앙값)
    baseline : float
        기준값 (기본: 0, 즉 상승/하락 판단)

    Returns
    -------
    float
        방향성 적중률 (0~1)
    """
    true_direction = np.sign(y_true - baseline)
    pred_direction = np.sign(y_pred - baseline)

    # 방향 일치
    correct = (true_direction == pred_direction).sum()

    return correct / len(y_true)

# ==================== 평가 함수 ====================

def evaluate_quantile_predictions(
    y_true: np.ndarray,
    y_q10: np.ndarray,
    y_q50: np.ndarray,
    y_q90: np.ndarray
) -> dict:
    """
    분위수 예측을 평가합니다.

    Parameters
    ----------
    y_true : np.ndarray
        실제 수익률
    y_q10, y_q50, y_q90 : np.ndarray
        예측 수익률 (분위수)

    Returns
    -------
    dict
        평가 결과
    """
    logger.info("\n평가 지표 계산 중...")

    # 중앙값 기준 오차
    mae_q50 = calculate_mae(y_true, y_q50)
    rmse_q50 = calculate_rmse(y_true, y_q50)
    mape_q50 = calculate_mape(y_true, y_q50)
    smape_q50 = calculate_smape(y_true, y_q50)

    # 범위 기반 평가
    coverage = calculate_coverage_rate(y_true, y_q10, y_q90)
    range_width = calculate_range_width(y_q10, y_q90)
    range_width_pct = calculate_range_width_pct(y_q10, y_q90, y_q50)

    # 방향성 평가
    directional_accuracy = calculate_directional_accuracy(y_true, y_q50)

    # 경고 메시지
    warnings = []

    if coverage < config.COVERAGE_THRESHOLD:
        warnings.append(f"⚠ Coverage가 낮습니다: {coverage*100:.1f}% (목표: {config.COVERAGE_THRESHOLD*100:.1f}%)")

    if range_width_pct > config.MAX_RANGE_WIDTH_PCT:
        warnings.append(f"⚠ 범위 폭이 넓습니다: {range_width_pct*100:.2f}% (기준: {config.MAX_RANGE_WIDTH_PCT*100:.2f}%)")

    # 결과 구성
    results = {
        'metrics': {
            'mae_q50': float(mae_q50),
            'rmse_q50': float(rmse_q50),
            'mape_q50': float(mape_q50),
            'smape_q50': float(smape_q50),
            'coverage_rate': float(coverage),
            'range_width': float(range_width),
            'range_width_pct': float(range_width_pct),
            'directional_accuracy': float(directional_accuracy),
        },
        'warnings': warnings,
        'timestamp': pd.Timestamp.now().isoformat()
    }

    # 로그 출력
    logger.info("\n=== 평가 결과 ===")
    logger.info(f"MAE (q50):               {mae_q50:.6f}")
    logger.info(f"RMSE (q50):              {rmse_q50:.6f}")
    logger.info(f"MAPE (q50):              {mape_q50*100:.2f}%")
    logger.info(f"sMAPE (q50):             {smape_q50*100:.2f}%")
    logger.info(f"\nCoverage Rate:           {coverage*100:.1f}%")
    logger.info(f"Range Width (절대):      {range_width:.6f}")
    logger.info(f"Range Width (상대):      {range_width_pct*100:.2f}%")
    logger.info(f"\nDirectional Accuracy:    {directional_accuracy*100:.1f}%")

    if warnings:
        logger.info("\n⚠ 경고:")
        for w in warnings:
            logger.info(f"  {w}")

    return results

# ==================== 시각화 함수 ====================

def plot_prediction_band(
    dates: pd.Series,
    actual: np.ndarray,
    q10: np.ndarray,
    q50: np.ndarray,
    q90: np.ndarray,
    output_path: Path = None
) -> None:
    """
    예측 범위와 실제값을 시각화합니다.

    Parameters
    ----------
    dates : pd.Series
        날짜
    actual : np.ndarray
        실제 수익률
    q10, q50, q90 : np.ndarray
        예측 분위수
    output_path : Path
        저장 경로
    """
    logger.info(f"\n예측 범위 차트 생성 중...")

    fig, ax = plt.subplots(figsize=config.FIGURE_SIZE_WIDE, dpi=config.FIGURE_DPI)

    # 날짜 배열
    dates_array = np.arange(len(actual))

    # 범위 표시
    ax.fill_between(dates_array, q10 * 100, q90 * 100, alpha=0.2, label='80% 범위 (q10~q90)', color='blue')
    ax.fill_between(dates_array, q10 * 100, q50 * 100, alpha=0.1, color='red')
    ax.fill_between(dates_array, q50 * 100, q90 * 100, alpha=0.1, color='green')

    # 선 표시
    ax.plot(dates_array, q50 * 100, label='예측 중앙값 (q50)', color='blue', linewidth=2)
    ax.plot(dates_array, q10 * 100, label='예측 하단 (q10)', color='red', linewidth=1, linestyle='--')
    ax.plot(dates_array, q90 * 100, label='예측 상단 (q90)', color='green', linewidth=1, linestyle='--')

    # 실제값 표시
    ax.scatter(dates_array, actual * 100, label='실제 수익률', color='black', s=20, alpha=0.6, zorder=5)

    ax.set_xlabel('거래일')
    ax.set_ylabel('수익률 (%)')
    ax.set_title('다음 거래일 종가 예측 범위')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
        logger.info(f"✓ 저장: {output_path}")

    plt.close()

def plot_actual_vs_predicted(
    dates: pd.Series,
    actual: np.ndarray,
    q50: np.ndarray,
    output_path: Path = None
) -> None:
    """
    실제값 vs 예측값을 비교합니다.

    Parameters
    ----------
    dates : pd.Series
        날짜
    actual : np.ndarray
        실제 수익률
    q50 : np.ndarray
        예측 중앙값
    output_path : Path
        저장 경로
    """
    logger.info(f"\n실제 vs 예측 차트 생성 중...")

    fig, ax = plt.subplots(figsize=config.FIGURE_SIZE_WIDE, dpi=config.FIGURE_DPI)

    dates_array = np.arange(len(actual))

    ax.plot(dates_array, actual * 100, label='실제 수익률', color='black', linewidth=2, marker='o', markersize=4)
    ax.plot(dates_array, q50 * 100, label='예측 수익률 (q50)', color='blue', linewidth=2, marker='s', markersize=4)

    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('거래일')
    ax.set_ylabel('수익률 (%)')
    ax.set_title('실제 vs 예측 수익률')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
        logger.info(f"✓ 저장: {output_path}")

    plt.close()

# ==================== 결과 저장 ====================

def save_evaluation_results(
    results: dict,
    output_dir: Path = config.OUTPUTS_DIR
) -> None:
    """
    평가 결과를 저장합니다.

    Parameters
    ----------
    results : dict
        평가 결과
    output_dir : Path
        출력 디렉토리
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON 저장
    json_path = output_dir / config.OUTPUT_EVALUATION_JSON
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"✓ 평가 결과 저장: {json_path}")

    # CSV 저장
    csv_path = output_dir / config.OUTPUT_EVALUATION_CSV
    df_results = pd.DataFrame([results['metrics']])
    df_results.to_csv(csv_path, index=False, encoding='utf-8')
    logger.info(f"✓ 평가 결과 저장: {csv_path}")

# ==================== 테스트 함수 ====================

if __name__ == "__main__":
    logger.info("evaluation 모듈 테스트 시작")

    # 더미 데이터
    np.random.seed(config.RANDOM_SEED)
    y_true = np.random.randn(100) * 0.02
    y_q10 = y_true - 0.01
    y_q50 = y_true + np.random.randn(100) * 0.005
    y_q90 = y_true + 0.01

    # 평가
    results = evaluate_quantile_predictions(y_true, y_q10, y_q50, y_q90)

    logger.info("\n✓ 평가 완료")
