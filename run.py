# -*- coding: utf-8 -*-
"""
run.py
======
프로젝트 CLI 엔트리포인트

사용법:
    python run.py --mode train
    python run.py --mode predict
    python run.py --mode evaluate
    python run.py --mode backtest
    python run.py --mode full

옵션:
    --mode: 실행 모드 (train, predict, evaluate, backtest, full)
    --start: 데이터 시작 날짜 (YYYY-MM-DD)
    --end: 데이터 종료 날짜 (YYYY-MM-DD)
    --ticker: 종목 코드 (기본: 005930)
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

# 프로젝트 루트 추가
sys.path.insert(0, str(Path(__file__).parent))

from app import config
from app.main import run_mode

def main():
    """메인 CLI 함수"""
    parser = argparse.ArgumentParser(
        description="삼성전자 종가 범위 예측 시스템",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예제:
  python run.py --mode train                              # 모델 학습
  python run.py --mode predict                            # 예측 실행
  python run.py --mode evaluate                           # 성능 평가
  python run.py --mode backtest                           # 백테스트
  python run.py --mode full                               # 전체 파이프라인
  python run.py --mode train --start 2020-01-01 --end 2024-12-31
        """
    )

    parser.add_argument(
        '--mode',
        type=str,
        default='train',
        choices=['train', 'predict', 'evaluate', 'backtest', 'full'],
        help='실행 모드 (기본값: train)'
    )

    parser.add_argument(
        '--start',
        type=str,
        default=config.DATA_START_DATE,
        help=f'데이터 시작 날짜 (YYYY-MM-DD, 기본값: {config.DATA_START_DATE})'
    )

    parser.add_argument(
        '--end',
        type=str,
        default=config.DATA_END_DATE,
        help=f'데이터 종료 날짜 (YYYY-MM-DD, 기본값: {config.DATA_END_DATE})'
    )

    parser.add_argument(
        '--ticker',
        type=str,
        default=config.TICKER,
        help=f'종목 코드 (기본값: {config.TICKER})'
    )

    args = parser.parse_args()

    # 날짜 형식 검증
    try:
        start_date = datetime.strptime(args.start, "%Y-%m-%d").strftime("%Y-%m-%d")
        end_date = datetime.strptime(args.end, "%Y-%m-%d").strftime("%Y-%m-%d")
    except ValueError:
        print("[ERROR] 날짜 형식이 잘못되었습니다. YYYY-MM-DD 형식을 사용하세요.")
        sys.exit(1)

    # 시작 날짜가 종료 날짜보다 늦지 않는지 확인
    if start_date > end_date:
        print("[ERROR] 시작 날짜가 종료 날짜보다 늦을 수 없습니다.")
        sys.exit(1)

    # 모드 실행
    print("\n" + "=" * 80)
    print("삼성전자 종가 범위 예측 시스템")
    print("=" * 80)
    print(f"모드: {args.mode}")
    print(f"종목: {args.ticker}")
    print(f"기간: {start_date} ~ {end_date}")
    print("=" * 80)

    try:
        run_mode(
            mode=args.mode,
            ticker=args.ticker,
            start_date=start_date,
            end_date=end_date
        )

        print("\n" + "=" * 80)
        print("[OK] 실행 완료")
        print("=" * 80)

    except KeyboardInterrupt:
        print("\n\n[ERROR] 사용자에 의해 중단되었습니다.")
        sys.exit(1)

    except Exception as e:
        print(f"\n\n[ERROR] 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
