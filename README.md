# Stock Range Predictor

## 📊 프로젝트 개요

이 프로젝트는 한국 주식 시장의 **다음 거래일 종가 범위(low, mid, high)**를 기계학습을 통해 예측하는 시스템입니다.
삼성전자(005930), SK하이닉스(000660), 한화에어로스페이스(012450) 등 **다양한 종목**을 지원합니다.

### 핵심 목표

**"내일 종가를 정확히 맞추는 것"이 아니라, "예측 구간이 의미 있는지 검증하는 것"**

이 시스템은 다음을 제공합니다:
- 데이터 기반의 종가 범위 예측
- 백테스트를 통한 성능 검증
- 투명한 평가 지표
- 확장 가능한 모듈 구조

---

## 🎯 왜 단일 가격 예측보다 범위 예측이 현실적인가?

### 단일 가격 예측의 문제점
- 금융 시장은 본질적으로 확률적(stochastic)
- "내일 종가 78,300원"이라는 점 추정은 거의 확률이 0에 가까움
- 실현된 모델 오차를 과장하기 쉬움

### 범위 예측의 장점
- **현실성**: "78,000~79,000원 범위"는 의미 있는 예측
- **검증 가능성**: Coverage rate로 정량화 가능
- **거래 가능성**: 범위를 활용한 거래 전략 수립 가능
- **불확실성 반영**: 예측 범위 폭이 시장 불확실성을 표현

### 이 프로젝트의 접근
```
수익률 분위수 예측 → 종가 범위 변환 → 백테스트 검증
(q10, q50, q90)      (low, mid, high)  (coverage, sharpe, etc.)
```

---

## 💾 설치 방법

### 1. 저장소 클론 또는 다운로드

```bash
git clone https://github.com/rlaxodn322/stock-range-predictor.git
cd stock-range-predictor
```

### 2. Python 환경 설정 (Python 3.11 권장)

```bash
# 가상 환경 생성 (선택 사항이지만 권장)
python -m venv venv

# 가상 환경 활성화
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### 3. 패키지 설치

```bash
pip install -r requirements.txt
```

### 4. 디렉토리 구조 확인

프로젝트 루트에 다음 디렉토리가 자동 생성됩니다:
- `models/` - 저장된 모델 파일
- `outputs/` - 예측 결과 및 평가 파일

---

## 🚀 실행 방법

### 기본 사용법

```bash
# 모델 학습
python run.py --mode train

# 다음 거래일 예측
python run.py --mode predict

# 성능 평가
python run.py --mode evaluate

# 백테스트
python run.py --mode backtest

# 전체 파이프라인 (학습 + 평가 + 백테스트 + 예측)
python run.py --mode full
```

### 고급 옵션

```bash
# 특정 기간의 데이터로 학습
python run.py --mode train --start 2020-01-01 --end 2024-12-31

# 다른 종목으로 실행
python run.py --mode train --ticker 000660           # SK하이닉스 학습
python run.py --mode full --ticker 012450            # 한화에어로스페이스 전체 파이프라인
python run.py --mode predict --ticker 051910         # 원하는 종목 예측
```

### 출력 파일

실행 후 `outputs/` 디렉토리에 다음 파일이 생성됩니다:

```
outputs/
├─ predictions.csv                  # 모든 예측 기록
├─ latest_prediction.json          # 최근 예측 결과
├─ evaluation_metrics.json         # 평가 지표
├─ feature_importance_q10.csv      # q10 모델 피처 중요도
├─ feature_importance_q50.csv      # q50 모델 피처 중요도
├─ feature_importance_q90.csv      # q90 모델 피처 중요도
├─ backtest_summary.json           # 백테스트 결과
├─ equity_curve.csv                # 자본금 변화 기록
├─ equity_curve.png                # 자본금 변화 차트
├─ prediction_band_plot.png        # 예측 범위 차트
└─ actual_vs_predicted.png         # 실제 vs 예측 비교
```

---

## 📁 파일 구조 설명

```
stock-range-predictor/
├─ app/
│  ├─ __init__.py                 # 패키지 초기화
│  ├─ config.py                   # 설정 및 상수 (모든 파라미터 관리)
│  ├─ data_loader.py              # 데이터 수집 및 전처리
│  ├─ features.py                 # 피처 엔지니어링
│  ├─ targets.py                  # 타깃 변수 생성
│  ├─ model.py                    # LightGBM 분위수 회귀 모델
│  ├─ train.py                    # 학습 파이프라인
│  ├─ predict.py                  # 예측 로직
│  ├─ evaluation.py               # 성능 평가 지표
│  ├─ backtest.py                 # 백테스트 엔진
│  ├─ utils.py                    # 유틸리티 함수
│  └─ main.py                     # 메인 조율 모듈
├─ models/                        # 저장된 모델 (자동 생성)
├─ outputs/                       # 결과 파일 (자동 생성)
├─ requirements.txt               # 패키지 의존성
├─ README.md                      # 이 파일
└─ run.py                         # CLI 엔트리포인트
```

---

## 📊 데이터 소스 설명

### 기본 데이터 소스: pykrx

```python
from pykrx import stock
df = stock.get_market_ohlcv('2024-01-01', '2024-12-31', '005930')  # 삼성전자
# 또는
df = stock.get_market_ohlcv('2024-01-01', '2024-12-31', '000660')  # SK하이닉스
```

**pykrx 특징:**
- 한국 주식 시장의 공식 데이터
- KRX(한국거래소) 기반
- 신뢰성 높음

### 보조 데이터 소스: yfinance (Fallback)

pykrx 실패 시 자동으로 yfinance로 전환:

```python
import yfinance as yf
df = yf.download('005930.KS', start='2024-01-01', end='2024-12-31')  # 삼성전자
# 또는
df = yf.download('000660.KS', start='2024-01-01', end='2024-12-31')  # SK하이닉스
```

**데이터 항목:**
- Date: 거래일
- Open: 시가
- High: 고가
- Low: 저가
- Close: 종가
- Volume: 거래량

---

## 🔧 예측 로직 설명

### 1단계: 수익률 분위수 예측

**개념:**
오늘 데이터를 기반으로 "내일 수익률"의 분포를 예측합니다.

**모델:** LightGBM Quantile Regression
- q10 (10분위): 하단 예상 수익률
- q50 (중앙값): 중심 수익률
- q90 (90분위): 상단 수익률

```python
# 예측 수익률 예시
q10_return = -0.011  # -1.1%
q50_return =  0.004  # +0.4%
q90_return =  0.013  # +1.3%
```

### 2단계: 종가 범위 환산

**계산식:**
```
low_close  = today_close × (1 + q10_return)
mid_close  = today_close × (1 + q50_return)
high_close = today_close × (1 + q90_return)
```

**예시:**
```
today_close = 78,000원
q10 = -0.011 → low  = 78,000 × 0.989 = 77,142원
q50 =  0.004 → mid  = 78,000 × 1.004 = 78,312원
q90 =  0.013 → high = 78,000 × 1.013 = 79,014원
```

### 3단계: 신호 생성

**신호 규칙:**
- **BULLISH**: q50 > 0.8% AND q10 > -0.5%
- **BEARISH**: q50 < -0.8% AND q90 < 0.3%
- **NEUTRAL**: 기타

### 피처 엔지니어링

**사용 피처 (총 50+ 개):**

1. **가격 기반 (8개)**
   - 일간 수익률, N일 수익률, 갭, 범위, 캔들 위치 등

2. **이동평균 (12개)**
   - 5/10/20/60일 MA, 비율, 기울기

3. **변동성 (9개)**
   - Rolling Volatility, ATR, 범위 등

4. **모멘텀 (6개)**
   - RSI, MACD, Stochastic

5. **거래량 (8개)**
   - 거래량 변화율, 상대 거래량 등

**중요 원칙:**
- 모든 피처는 **"오늘까지 알 수 있는 정보"**만 사용
- 미래 데이터 누수(look-ahead bias) 철저히 방지
- 결측치는 초반 롤링 윈도우 계산으로 인해 자연스럽게 제거

---

## 📈 백테스트 로직 설명

### 백테스트 규칙 (A안)

**진입:**
- 오늘 데이터로 내일 예측 수행
- 다음 거래일 시가에 진입 (시뮬레이션상 현재 close 사용)
- 진입 신호: q50 > 0.5% AND q10 > -0.5%

**청산:**
- 같은 거래일 종가에 청산
- 청산 가격 = 진입가 × (1 + 실제 수익률)

**거래 비용:**
- 왕복 거래 비용: 0.1% (수수료 + 슬리피지)

### 성과 지표

```
총 거래 수:   백테스트 기간 동안의 거래 횟수
승률:         (승리 거래 수) / (총 거래 수) × 100%

총 수익률:    (최종 자본금 - 초기 자본금) / 초기 자본금 × 100%
CAGR:         연간화 수익률 (복합 연간 성장률)

MDD:          최대 낙폭 (최악의 시점의 손실률)
Sharpe Ratio: 수익률 대비 위험도 지표 (높을수록 좋음)

Profit Factor: (승리 거래 총액) / (패배 거래 총액)
```

### 백테스트 vs Buy & Hold

```
Buy & Hold 기준선: 매수 후 보유
전략 수익률 > Buy & Hold 수익률인지 확인
```

---

## 📊 평가 지표 설명

### 1. 오차 지표 (중앙값 q50 기준)

```
MAE  = 평균절대오차 (Mean Absolute Error)
       작을수록 좋음

RMSE = 루트평균제곱오차 (Root Mean Squared Error)
       이상치에 더 민감함

MAPE = 평균절대백분율오차 (Mean Absolute Percentage Error)
       상대적 오차율
```

### 2. 범위 평가

```
Coverage Rate = (실제값이 범위 안에 든 비율)
                최소 90% 이상이 목표

Range Width = 예측 범위의 폭
              (high - low) / mid
              좁을수록 정확하지만, 너무 좁으면 신뢰도 하락
```

### 3. 방향성 평가

```
Directional Accuracy = 상승/하락 판단 정확도
                       mid > 0 또는 < 0으로 판단
```

### 4. 경고 시스템

```
⚠ Coverage < 90%     → 범위가 좁음, 신뢰도 낮음
⚠ Range Width > 5%   → 범위가 너무 넓음, 정보성 낮음
⚠ 낮은 방향성 정확도 → 추세 예측 능력 부족
```

---

## ⚠️ 한계점 및 주의사항

### 1. 금융 시장 예측의 근본적 한계

```
"과거 패턴이 미래를 보장하지 않습니다"

주식 가격은:
- 정보의 도착 (뉴스, 실적 등)에 반응
- 시장 심리, 공포/탐욕에 영향받음
- 구조적 변화(정책, 경제 사이클)를 반영

따라서 ML 모델의 예측력에는 한계가 있습니다.
```

### 2. 과적합(Overfitting) 위험

```
해결책:
- 시계열 데이터 순서 유지 (랜덤 셔플 금지)
- Train/Val/Test 시간 기반 분리
- Early Stopping으로 과적합 방지
- 검증 데이터로 성능 감시
```

### 3. 외부 변수 미반영

```
미반영 요인:
- 뉴스, 공시, 실적 발표
- 정책 변화, 금리 인상
- 환율, 글로벌 시장 영향
- 기술적 이슈, 공급망 변화

개선안:
- 뉴스 감성분석 추가
- 거시경제 지표 통합
- 실시간 옵션 시장 데이터
```

### 4. 데이터 누수 방지

```
이 프로젝트에서 시행한 조치:

✓ shift(-1) 사용으로 "다음 거래일" 정보 분리
✓ 피처는 "오늘까지"의 정보만 사용
✓ 타깃은 다음 거래일 실제값
✓ 마지막 행 자동 제거 (다음 거래일 없음)
✓ 명시적 look-ahead bias 검증
```

---

## ⚖️ 투자 권고 및 책임 한계

### ⚠️ **중요한 경고**

```
이 시스템은 교육 및 연구 목적으로 개발되었습니다.

절대로 다음을 보장하지 않습니다:
- 미래 가격 정확한 예측
- 수익성 보장
- 손실 방지

사용자의 투자 결정은:
- 전적으로 사용자의 책임
- 충분한 조사 후 결정
- 위험 감수 능력 범위 내에서만
- 전문가 자문 권장
```

### 법적 책임 부인

```
본 프로젝트 개발자 및 배포자는:
- 예측의 정확성에 대한 책임 없음
- 투자 손실에 대한 책임 없음
- 법적, 재무적 조언 제공 안 함
- 데이터 오류에 대한 책임 제한적
```

---

## 🔮 향후 확장 아이디어

### 1. 데이터 확장
- [ ] 포트폴리오: 여러 종목 동시 추적
- [ ] 코스피/코스닥 지수 통합
- [ ] 환율(USD/KRW) 영향도
- [ ] 외국인/기관 순매수 (주중심)
- [ ] 미국 반도체 지수(SOX)
- [ ] VIX 공포 지수
- [ ] 업종별 상대 강도(Relative Strength)

### 2. 모델 고도화
- [ ] LSTM/GRU 시계열 모델
- [ ] Transformer 어텐션 메커니즘
- [ ] Ensemble (여러 모델 조합)
- [ ] Bayesian Optimization 하이퍼파라미터
- [ ] 부스팅 앙상블 (XGBoost, CatBoost)

### 3. 특성 개선
- [ ] 뉴스 감성분석 (NLP)
- [ ] 기술적 패턴 인식
- [ ] 거래 심도(Order Book) 분석
- [ ] 옵션 시장 정보
- [ ] 머신러닝 기반 계절성 감지

### 4. 백테스트 고도화
- [ ] Walk-Forward Validation (자동 재학습)
- [ ] 동적 포지션 사이징
- [ ] 손절매(Stop Loss) 규칙
- [ ] 포지션 관리 개선
- [ ] 다중 전략 비교

### 5. 운영 체계
- [ ] 실시간 예측 API화
- [ ] 웹 대시보드
- [ ] 모바일 알림
- [ ] 자동 거래 시스템(자동 주문)
- [ ] 성과 모니터링 시스템

### 6. 코드 품질
- [ ] 단위 테스트 추가
- [ ] 통합 테스트
- [ ] CI/CD 파이프라인
- [ ] Docker 컨테이너화
- [ ] 문서화 강화

---

## 🛠️ 문제 해결

### 데이터 수집 오류

```
오류: "pykrx에서 데이터를 찾을 수 없습니다"

해결책:
1. 네트워크 연결 확인
2. KRX 서버 상태 확인 (영업 시간 확인)
3. 날짜 범위 확인 (충분한 과거 데이터 필요)
4. Fallback (yfinance)가 자동 실행됨
```

### 모델 로드 실패

```
오류: "모델 파일을 찾을 수 없습니다"

해결책:
1. 먼저 학습 모드 실행: python run.py --mode train --ticker 005930
2. models/ 디렉토리 확인
3. 모델 파일 존재 확인:
   - models/{TICKER}_model_q10.pkl
   - models/{TICKER}_model_q50.pkl
   - models/{TICKER}_model_q90.pkl

   예: 삼성전자(005930)인 경우
   - models/005930_model_q10.pkl
   - models/005930_model_q50.pkl
   - models/005930_model_q90.pkl
```

### 메모리 부족

```
오류: MemoryError

해결책:
1. 데이터 기간 축소: --start, --end 옵션 사용
2. 배치 크기 감소 (config.py의 LGBM_PARAMS 수정)
3. 메모리 여유가 있는 시스템에서 실행
```

---

## 📚 참고 자료

### 라이브러리 문서
- [pandas](https://pandas.pydata.org/docs/) - 데이터 조작
- [scikit-learn](https://scikit-learn.org/) - 머신러닝
- [LightGBM](https://lightgbm.readthedocs.io/) - Gradient Boosting
- [pykrx](https://github.com/sharpe31/pykrx) - 한국 주식 데이터
- [yfinance](https://github.com/ranaroussi/yfinance) - 글로벌 금융 데이터

### 금융 분석 개념
- [Quantile Regression](https://en.wikipedia.org/wiki/Quantile_regression) - 분위수 회귀
- [Feature Engineering](https://en.wikipedia.org/wiki/Feature_engineering) - 피처 엔지니어링
- [Backtesting](https://en.wikipedia.org/wiki/Backtesting) - 백테스트
- [Time Series Analysis](https://en.wikipedia.org/wiki/Time_series) - 시계열 분석

---

## 📝 라이선스

```
MIT License - 자유롭게 사용, 수정, 배포 가능
단, 저자 표시 필요
```

---

## 💬 문의 및 피드백

```
버그 보고 또는 개선 아이디어가 있으시면:
- 이슈 등록
- 풀 리퀘스트 제출
- 의견 제안
```

---

## 🎓 마지막 당부

```
"과거 성과가 미래 성과를 보장하지 않습니다"

이 프로젝트는:
✓ 금융 데이터 분석의 기초 학습
✓ 머신러닝 시계열 예측의 실전 경험
✓ 백테스트의 중요성 이해

에 도움이 될 것입니다.

하지만 실제 투자는 신중하게, 전문가 조언을 받아 진행하세요.

Happy Learning! 🚀
```

---

**최종 수정**: 2026-03-09
**버전**: 1.0.0
**프로젝트**: Stock Range Predictor
**지원 종목**: 한국 주식 전체 (pykrx 지원 종목)
