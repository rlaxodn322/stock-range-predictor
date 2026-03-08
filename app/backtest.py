"""
backtest.py
===========
예측 범위 기반 단순 백테스트를 구현합니다.

백테스트 규칙 (A안):
- 오늘 데이터로 다음 거래일 예측 수행
- 다음 거래일 시가에 진입
- 다음 거래일 종가에 청산
- 거래비용 적용

성과 지표:
- Cumulative Return
- CAGR
- Max Drawdown
- Sharpe Ratio
- Win Rate
- Total Trades
- Profit Factor
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Tuple
import json
import matplotlib.pyplot as plt

from . import config

logger = config.setup_logger(__name__)

# ==================== 백테스트 엔진 ====================

class SimpleRangeBacktester:
    """
    범위 예측 기반 단순 백테스트 엔진
    """

    def __init__(
        self,
        initial_capital: float = 1000000,
        position_size: float = 1.0,
        transaction_cost: float = config.TRANSACTION_COST_PCT
    ):
        """
        Parameters
        ----------
        initial_capital : float
            초기 자본금
        position_size : float
            매수 수량 비율 (1.0 = 자본금 전액)
        transaction_cost : float
            거래 비용 (%)
        """
        self.initial_capital = initial_capital
        self.position_size = position_size
        self.transaction_cost = transaction_cost

        self.trades = []
        self.equity_curve = []
        self.positions = []

    def generate_signals(
        self,
        df: pd.DataFrame,
        predictions: Dict[str, np.ndarray]
    ) -> pd.DataFrame:
        """
        거래 신호를 생성합니다.

        신호 규칙:
        - q50 > entry_threshold_q50 AND q10 > entry_threshold_q10: BUY
        - 기타: HOLD (또는 SELL)

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV 데이터
        predictions : Dict[str, np.ndarray]
            분위수 예측 {'q10': array, 'q50': array, 'q90': array}

        Returns
        -------
        pd.DataFrame
            신호가 추가된 데이터
        """
        df = df.copy()

        q10 = predictions['q10']
        q50 = predictions['q50']
        q90 = predictions['q90']

        # 진입 신호
        buy_condition = (
            (q50 > config.BACKTEST_ENTRY_THRESHOLD_Q50) &
            (q10 > config.BACKTEST_ENTRY_THRESHOLD_Q10)
        )

        df['signal'] = np.where(buy_condition, 1, 0)  # 1: BUY, 0: HOLD/SELL

        buy_count = (df['signal'] == 1).sum()
        logger.info(f"✓ 신호 생성 완료: {buy_count}개의 BUY 신호")

        return df

    def backtest(
        self,
        df: pd.DataFrame,
        signals: np.ndarray,
        next_returns: np.ndarray
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        백테스트를 실행합니다.

        백테스트 구조:
        1. 신호가 1이면 다음 거래일 시가에 진입 (시뮬레이션상 현재 close로 설정)
        2. 같은 거래일 종가에 청산 (다음 거래일 return으로 계산)
        3. 거래비용 차감

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV 데이터
        signals : np.ndarray
            거래 신호 (1 또는 0)
        next_returns : np.ndarray
            다음 거래일 수익률

        Returns
        -------
        Tuple[pd.DataFrame, Dict]
            (결과 데이터, 성과 지표)
        """
        logger.info("\n백테스트 실행 중...")

        equity = self.initial_capital
        position_count = 0
        winning_trades = 0
        losing_trades = 0
        total_pnl = 0.0

        trade_list = []
        equity_list = [equity]

        for i in range(len(df) - 1):
            # 진입 신호
            if signals[i] == 1:
                # 진입 가격: 현재(i)의 종가
                entry_price = df.iloc[i]['Close']
                entry_qty = (equity * self.position_size) / entry_price

                # 다음 거래일(i+1) 수익률 적용
                return_rate = next_returns[i]

                # 청산 가격: 진입가 * (1 + return_rate)
                exit_price = entry_price * (1 + return_rate)

                # 손익 계산 (거래비용 포함)
                gross_pnl = (exit_price - entry_price) * entry_qty
                transaction_cost_amount = equity * self.transaction_cost
                net_pnl = gross_pnl - transaction_cost_amount

                # 자본금 업데이트
                equity += net_pnl

                # 거래 기록
                trade = {
                    'date': df.iloc[i]['Date'],
                    'entry_date': df.iloc[i]['Date'],
                    'exit_date': df.iloc[i + 1]['Date'],
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'qty': entry_qty,
                    'return': return_rate,
                    'pnl': net_pnl,
                    'equity': equity
                }
                trade_list.append(trade)

                # 승패 기록
                if net_pnl > 0:
                    winning_trades += 1
                else:
                    losing_trades += 1

                total_pnl += net_pnl
                position_count += 1

            equity_list.append(equity)

        # 결과 데이터프레임
        df_result = df.copy()
        df_result['signal'] = signals
        df_result['next_return'] = next_returns
        # equity_list는 len(df) 크기이므로 그대로 사용
        df_result['equity'] = equity_list[:len(df)]

        # 성과 지표 계산
        total_trades = position_count
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
        avg_win = total_pnl / winning_trades if winning_trades > 0 else 0.0
        avg_loss = -total_pnl / losing_trades if losing_trades > 0 else 0.0
        profit_factor = abs(winning_trades * avg_win / (losing_trades * avg_loss)) if losing_trades > 0 and avg_loss != 0 else 0.0

        # 누적 수익률
        total_return = (equity - self.initial_capital) / self.initial_capital
        total_return_pct = total_return * 100

        # CAGR (복합 연간 수익률)
        trading_days = len(df)
        years = trading_days / 252  # 연간 거래일수 252
        if years > 0 and equity > 0:
            cagr = (equity / self.initial_capital) ** (1 / years) - 1
        else:
            cagr = 0.0

        # MDD (최대 낙폭)
        cummax = np.maximum.accumulate(equity_list)
        drawdowns = (np.array(equity_list) - cummax) / cummax
        mdd = np.min(drawdowns)

        # Sharpe Ratio (더미 계산)
        if len(df_result) > 0:
            daily_returns = df_result['next_return'].values
            if np.std(daily_returns) > 0:
                sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)
            else:
                sharpe_ratio = 0.0
        else:
            sharpe_ratio = 0.0

        performance = {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': float(win_rate),
            'avg_trade_pnl': float(total_pnl / total_trades) if total_trades > 0 else 0.0,
            'total_pnl': float(total_pnl),
            'total_return': float(total_return),
            'total_return_pct': float(total_return_pct),
            'cagr': float(cagr),
            'mdd': float(mdd),
            'sharpe_ratio': float(sharpe_ratio),
            'profit_factor': float(profit_factor),
            'final_equity': float(equity),
            'initial_capital': float(self.initial_capital),
        }

        logger.info(f"✓ 백테스트 완료")
        logger.info(f"\n=== 백테스트 결과 ===")
        logger.info(f"총 거래 수:           {total_trades}")
        logger.info(f"승리 거래:           {winning_trades}")
        logger.info(f"패배 거래:           {losing_trades}")
        logger.info(f"승률:                {win_rate*100:.1f}%")
        logger.info(f"\n총 손익:             {total_pnl:,.0f}원")
        logger.info(f"총 수익률:           {total_return_pct:.2f}%")
        logger.info(f"CAGR:                {cagr*100:.2f}%")
        logger.info(f"최대 낙폭 (MDD):     {mdd*100:.2f}%")
        logger.info(f"Sharpe Ratio:        {sharpe_ratio:.3f}")
        logger.info(f"\n최종 자본금:         {equity:,.0f}원")

        return df_result, trade_list, performance

    def plot_equity_curve(
        self,
        df_result: pd.DataFrame,
        equity_list: list,
        output_path: Path = None
    ) -> None:
        """
        Equity curve를 시각화합니다.

        Parameters
        ----------
        df_result : pd.DataFrame
            백테스트 결과
        equity_list : list
            자본금 변화
        output_path : Path
            저장 경로
        """
        logger.info(f"\nEquity curve 차트 생성 중...")

        fig, ax = plt.subplots(figsize=config.FIGURE_SIZE_WIDE, dpi=config.FIGURE_DPI)

        ax.plot(np.arange(len(equity_list)), equity_list, label='Equity Curve', linewidth=2, color='blue')
        ax.axhline(y=self.initial_capital, color='red', linestyle='--', label='Initial Capital', linewidth=1)

        ax.set_xlabel('거래일')
        ax.set_ylabel('자본금 (원)')
        ax.set_title('Equity Curve')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M'))

        plt.tight_layout()

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
            logger.info(f"✓ 저장: {output_path}")

        plt.close()

# ==================== 백테스트 파이프라인 ====================

def run_backtest(
    df: pd.DataFrame,
    predictions: Dict[str, np.ndarray],
    initial_capital: float = 1000000,
    output_dir: Path = config.OUTPUTS_DIR
) -> Tuple[Dict, pd.DataFrame]:
    """
    전체 백테스트 파이프라인을 실행합니다.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV 데이터 (Date, Close 포함)
    predictions : Dict[str, np.ndarray]
        분위수 예측
    initial_capital : float
        초기 자본금
    output_dir : Path
        출력 디렉토리

    Returns
    -------
    Tuple[Dict, pd.DataFrame]
        (성과 지표, 결과 데이터)
    """
    logger.info("=" * 80)
    logger.info("백테스트 파이프라인 시작")
    logger.info("=" * 80)

    # 백테스터 생성
    backtester = SimpleRangeBacktester(initial_capital=initial_capital)

    # 신호 생성
    df_with_signals = backtester.generate_signals(df, predictions)
    signals = df_with_signals['signal'].values

    # 다음 거래일 수익률 계산 (실제 값)
    next_returns = (df['Close'].shift(-1) / df['Close'] - 1).values

    # 백테스트 실행
    df_result, trade_list, performance = backtester.backtest(df_with_signals, signals, next_returns)

    # 결과 저장
    output_dir.mkdir(parents=True, exist_ok=True)

    # Performance 저장
    perf_path = output_dir / config.OUTPUT_BACKTEST_SUMMARY_JSON
    with open(perf_path, 'w', encoding='utf-8') as f:
        json.dump(performance, f, ensure_ascii=False, indent=2)
    logger.info(f"✓ 백테스트 결과 저장: {perf_path}")

    # Equity curve 저장
    equity_csv_path = output_dir / config.OUTPUT_EQUITY_CURVE_CSV
    df_equity = pd.DataFrame({
        'date': df_result['Date'].values,
        'equity': df_result['equity'].values
    })
    df_equity.to_csv(equity_csv_path, index=False, encoding='utf-8')
    logger.info(f"✓ Equity curve CSV 저장: {equity_csv_path}")

    # Equity curve 차트
    equity_chart_path = output_dir / config.OUTPUT_EQUITY_CURVE_PNG
    backtester.plot_equity_curve(df_result, df_result['equity'].tolist(), equity_chart_path)

    logger.info("\n" + "=" * 80)
    logger.info("백테스트 파이프라인 완료")
    logger.info("=" * 80)

    return performance, df_result

# ==================== 테스트 함수 ====================

if __name__ == "__main__":
    logger.info("backtest 모듈 테스트 시작")

    # 더미 데이터
    np.random.seed(config.RANDOM_SEED)
    dates = pd.date_range('2023-01-01', periods=252)
    close_prices = 50000 * np.exp(np.cumsum(np.random.randn(252) * 0.01))

    df = pd.DataFrame({
        'Date': dates,
        'Open': close_prices * 0.99,
        'High': close_prices * 1.01,
        'Low': close_prices * 0.98,
        'Close': close_prices,
        'Volume': np.random.randint(1000000, 5000000, 252)
    })

    predictions = {
        'q10': np.random.randn(252) * 0.01 - 0.005,
        'q50': np.random.randn(252) * 0.01,
        'q90': np.random.randn(252) * 0.01 + 0.005
    }

    performance, df_result = run_backtest(df, predictions, initial_capital=1000000)

    logger.info("\n✓ 백테스트 테스트 완료")
