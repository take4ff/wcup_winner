# サッカー国際試合 予測モデル・シミュレータ

本プロジェクトは、過去の国際Aマッチ結果を用いて国ごとの実力を評価するモデルを構築し、FIFAワールドカップをテストデータとしたバックテスト評価、大会全体のモンテカルロ・シミュレーション（優勝確率算出）、および2026年大会のリアルタイム予測・ベッティング推奨を行う総合システムです。

---

## 開発環境とセットアップ

本プロジェクトは以下の環境で動作確認・構築されています。

* **Pythonバージョン:** `3.10.19`
* **ハードウェア要件:** GPU不要（標準的なマルチコアCPUのみの環境で十分に高速動作します）
* **主要な依存ライブラリとバージョン:**
  * `pandas` == `2.3.3`
  * `numpy` == `2.2.5`
  * `scikit-learn` == `1.7.2`
  * `lightgbm` == `4.6.0`
  * `optuna` == `4.9.0`
  * `scipy` == `1.15.3`
  * `joblib` == `1.5.3`
  * `requests` == `2.32.5`
  * `sqlalchemy` == `2.0.50`
  * `beautifulsoup4` == `4.14.3`

### 💻 環境の構築・再現手順

リポジトリ直下にある [environment.yml](./environment.yml) を使用して、Conda環境を簡単に再現できます。

**1. Conda環境の新規作成:**
```bash
conda env create -f environment.yml -n wcup_winner
# macOS (Apple Silicon/Intel) で lightgbm が libomp.dylib エラーになる場合:
conda install -n wcup_winner -c conda-forge llvm-openmp -y
```

**2. 環境のアクティベート:**
```bash
conda activate wcup_winner
```

**3. （参考）既存環境への一括インストール:**
既存の環境にライブラリのみをインストールする場合は、以下を実行してください。
```bash
pip install pandas==2.3.3 numpy==2.2.5 scikit-learn==1.7.2 lightgbm==4.6.0 optuna==4.9.0 scipy==1.15.3 joblib==1.5.3 requests==2.32.5 sqlalchemy==2.0.50 beautifulsoup4==4.14.3
```

---

## ファイル構成

```text
wcup_winner/
├── README.md
├── src/
│   ├── pipeline/                      # コアパイプライン（データ収集〜学習）
│   │   ├── download_data.py           # 国際Aマッチデータの自動収集
│   │   ├── elo.py                     # 全試合からElo Ratingを時系列算出
│   │   ├── preprocess.py              # 市場価値結合・Elo調整rolling特徴量の生成
│   │   └── train_model.py             # Poisson + LightGBM(Optuna) アンサンブルモデル学習
│   ├── odds/                          # オッズデータ準備
│   │   ├── prepare_odds.py            # 2022年カタールW杯オッズ
│   │   ├── prepare_odds_2018.py       # 2018年ロシアW杯オッズ（全64試合手動定義）
│   │   ├── prepare_odds_2026_groups.py# 2026年グループステージオッズ（実値+推定値）
│   │   └── fetch_odds_2026.py         # The-Odds-APIによる実オッズの自動取得
│   ├── backtest/                      # バックテスト・検証
│   │   ├── backtest.py                # --year 2018/2022 で大会切替
│   │   └── walkforward.py             # 時系列ウォークフォワード検証・パラメータ調整
│   └── predict/                       # 予測・シミュレーション・推奨
│       ├── simulator.py               # 2022年大会 32カ国モンテカルロ
│       ├── simulator_2026.py          # 2026年大会 48カ国対応モンテカルロ
│       ├── predict_scores_2026.py     # 全72試合 スコア予測・勝敗確率出力
│       ├── bet_advisor_2026.py        # EV計算・ケリー基準ベッティング推奨
│       ├── calculate_winner_ev_general.py # WINNER 18択期待値計算 (汎用ツール)
│       └── settle_bets_2026.py        # WINNER 自動精算・ROI集計ツール
├── data/
│   ├── buy_status.csv                 # WINNER 購入履歴・結果ステータス管理
│   ├── raw/
│   │   ├── match/                     # 試合結果データ
│   │   │   ├── results.csv            # 全国際Aマッチ結果（download_data.pyで随時更新）
│   │   │   └── shootouts.csv          # PK戦結果
│   │   ├── odds/                      # オッズデータ
│   │   │   ├── odds_qatar2022.csv
│   │   │   ├── odds_russia2018.csv
│   │   │   ├── odds_groups_2026.csv   # ★試合前に fetch_odds_2026.py で実オッズに更新
│   │   │   └── winner_inputs/         # WINNER 18択オッズの手入力CSV
│   │   └── squad/                     # 選手総市場価値データ（大会世代別）
│   │       ├── squad_values_2018.csv
│   │       ├── squad_values_2022.csv
│   │       ├── squad_values_2026.csv
│   │       └── squad_penalties_2026.csv # ★主力離脱時に team,multiplier,reason を記入（例: 0.85）
│   └── processed/
│       ├── features.csv               # 特徴量（全大会共通）
│       ├── results_with_elo.csv       # Elo付き試合結果
│       ├── 2018/                      # 2018年大会出力
│       │   ├── backtest_results.csv
│       │   ├── backtest_metrics.csv
│       │   └── roi_summary.csv
│       ├── 2022/                      # 2022年大会出力
│       │   ├── backtest_results.csv
│       │   ├── backtest_metrics.csv
│       │   ├── roi_summary.csv
│       │   └── simulation_results.csv
│       └── 2026/                      # 2026年大会出力
│           ├── simulation_results.csv
│           ├── predicted_scores.csv
│           ├── bet_recommendations.csv
│           └── winner_matches/        # WINNER個別試合の期待値CSV（自動作成）
└── models/
    ├── poisson_model.joblib
    ├── lgbm_model.joblib
    ├── lgbm_classifier_model.joblib
    └── feature_cols.joblib
```


---

## 実行手順

### 基本パイプライン（データ更新 → 学習 → 予測）

```bash
# 1. データ収集（最新の試合結果を取得）
python src/download_data.py

# 2. Elo Rating の時系列計算
python src/elo.py

# 3. 特徴量の生成
python src/preprocess.py

# 4. 予測モデルの学習・保存（Poisson + LightGBM, Optunaチューニング）
# 本番用: 全データで学習 → models/ に保存（デフォルト）
python src/pipeline/train_model.py
# バックテスト用: 大会開幕前まで学習し専用ディレクトリに保存（本番モデルと共存できる）
python src/pipeline/train_model.py --train_end 2018-06-14 --model_dir models/backtest_2018
python src/pipeline/train_model.py --train_end 2022-11-20 --model_dir models/backtest_2022
```

### バックテスト（2018年 / 2022年大会）

バックテストは `models/backtest_{year}/` が存在すれば自動でそれを使用します（リーク防止）。
無い場合は本番モデルにフォールバックし、警告を表示します。

```bash
# 2022年カタール大会
python src/odds/prepare_odds.py
python src/backtest/backtest.py --year 2022

# 2018年ロシア大会
python src/odds/prepare_odds_2018.py
python src/backtest/backtest.py --year 2018
```

### ウォークフォワード検証（パラメータ調整）

W杯2大会(128試合)では差が小さいパラメータはノイズに埋もれるため、
全国際試合で「その年より前のデータで学習 → その年を予測」を繰り返して大サンプルで評価します。

```bash
python src/backtest/walkforward.py --mode probs   # Dixon-Coles ρ × 分類器ブレンド比
python src/backtest/walkforward.py --mode shrink  # 市場シュリンク比（probsの後に実行）
python src/backtest/walkforward.py --mode elo     # Elo Kスケール × ホーム補正
```

### 2026年大会 予測・シミュレーション

```bash
# グループステージ全72試合のスコア予測（1X2確率は Poisson行列×分類器のブレンド, --cls_blend で調整可）
python src/predict/predict_scores_2026.py

# 48カ国 優勝確率シミュレーション（1万回モンテカルロ）
python src/predict/simulator_2026.py

# ベッティング推奨（EV・ケリー計算）
python src/odds/prepare_odds_2026_groups.py # 推定オッズCSVの初期生成
python src/odds/fetch_odds_2026.py          # APIから最新の実オッズをダウンロードして更新
# --market_blend: モデル確率を市場確率へシュリンクする比率（EV過大評価の抑制, default 0.3）
# --max_ev: サニティ上限。超過分はデータ異常の疑いとして警告・除外（default 2.0）
python src/predict/bet_advisor_2026.py --ev_thresh 1.05 --kelly half --bankroll 100

# WINNERの個別試合 18択期待値の計算
# (1) オッズ入力用テンプレートの生成（初回のみ。生成したCSVのodds列にWINNERのオッズを記入する）
python src/predict/calculate_winner_ev_general.py --generate_template data/raw/odds/winner_inputs/03_winner_xxx_yyy.csv
# (2) CSV入力で計算 → data/processed/2026/winner_matches/<連番>_winner_<home>_<away>_ev.csv に出力
python src/predict/calculate_winner_ev_general.py --home Mexico --away "South Africa" \
    --odds_csv data/raw/odds/winner_inputs/01_winner_mexico_south_africa.csv
# (3) または対話型でオッズを直接入力
python src/predict/calculate_winner_ev_general.py --home Mexico --away "South Africa"
# ※ λ（期待得点）は predicted_scores.csv から読むため、モデル更新後は predict_scores_2026.py を先に実行すること

# WINNER購入結果の自動精算・ROI集計
python src/predict/settle_bets_2026.py
```

---

## モデルアーキテクチャ

### 予測モデル（アンサンブル）

| モデル | 手法 | 役割 |
|:---|:---|:---|
| Poisson Regression | Ridge正則化 Poisson | 期待得点（λ）の線形推定 |
| LightGBM (Regressor) | Poisson目的関数 + Optuna CV最適化 | 非線形特徴量から期待得点を推定 |
| LightGBM (Classifier) | 3クラス分類 (H/D/A) | 勝敗確率の直接推定 |
| **期待得点 Ensemble** | (Poisson + LGBM) / 2 | 安定した期待得点を出力 |
| **1X2確率 Blend** | 0.75×Dixon-Coles行列 + 0.25×分類器 | 最終的な勝敗確率（バックテスト・2026年予測で共通） |

> ブレンド比0.25とDixon-Coles ρ=-0.09 は、ウォークフォワード検証（2019〜2026年の全国際試合 7,183試合で
> 「その年より前のデータのみで学習→その年を予測」）のLog Loss最小値（`--cls_blend` で変更可）。
> ベッティング推奨時はさらに、マージン除去済みの市場インプライド確率と混合（`--market_blend`, デフォルト0.5）してEVの過大評価を抑制する。

### モンテカルロ・シミュレーション（2026年）

- **FIFA公式ノックアウトブラケット**を実装（R32の16試合スロット＋ベスト3位の許容グループ制約を
  バックトラッキングで充足。全495通りの3位組み合わせで検証済み）
- ノックアウトの同点時は **延長戦（λ×1/3のPoisson）→ PK戦（50/50）** で決着
- 開催国（米加墨）の自国開催試合には `was_home=1` を適用
- Elo の K値はウォークフォワード評価により旧値×1.2 に調整（ホーム補正は+100を維持）

### 主要特徴量（93個）

- **Elo Rating差** (試合前時点、対戦相手調整済み)
- **選手総市場価値差** (squad_value_diff) と **欠損フラグ** (squad_value_missing, 欠損時は50M€で補完)
- **直近n試合rolling統計** (得点/失点/勝率 × roll5/roll10/ewm5/ewm10 × 全試合/公式戦)
- **休養日数** (rest_days, 前試合からの日数・上限30日)
- **W杯出場経験** (last_wcup_matches, 2022年大会まで反映)
- **実質ホームアドバンテージ** (same_confederation / is_host, 全FIFA加盟国+歴史的代表をカバーする連盟辞書に基づく)
- **ホームゲーム識別フラグ** (was_home)

### Dixon-Coles補正

同時確率行列の低得点セル（0-0, 1-0, 0-1, 1-1）に補正パラメータ $\rho = -0.03$ を適用し、引き分け確率の過小推定を解消します。

---

## 主要な成果

※ 数値は特徴量バグ修正・新特徴量（休養日数・市場価値欠損フラグ）・調整済みパラメータ
（Elo K×1.2, 1X2ブレンド0.25）による 2026-06-12 再計測。**両大会とも開幕前カットオフのモデル**
（2018年: 2018-06-14 / 2022年: 2022-11-20, `models/backtest_{year}/`）を使用したリークなしの評価。

### バックテスト比較 (2018年 vs 2022年)

| 指標 | 2018年 ロシア大会 | 2022年 カタール大会 |
|:---|:---:|:---:|
| **平均 Log Loss** | **0.95790** | 1.04704 |
| **平均 Brier Score** | **0.56900** | 0.61282 |
| **定額ベット ROI (EV>1.20)** | 141.67% | **278.79%** |
| **Half-Kelly 最終資金** | 263.07 | **446.43** |

> 2018年大会はLog Loss・Brier Score共に優秀。2022年大会はアップセット（日本のドイツ/スペイン撃破等）が多かった分、確率予測の難易度が高かった。
> 2018年は学習データが2015〜2018年の約3年分しかないため、ベットROIは2022年より控えめ。
> 大サンプルでの較正評価はウォークフォワード検証を参照（2019-2026年 7,183試合で Log Loss 0.857）。

### 2022年カタール大会 バックテスト詳細

#### A. 定額ベット戦略 (1.0ユニット)
| EV閾値 | ベット数 | 回収率 |
|:---:|:---:|:---:|
| EV > 1.00 | 60 | **202.23%** |
| EV > 1.05 | 51 | **214.82%** |
| EV > 1.10 | 42 | **231.36%** |
| EV > 1.15 | 35 | **262.86%** |
| EV > 1.25 | 29 | **291.38%** |

#### B. ケリー基準動的ベット戦略 (初期資金100.0)
| 戦略 | 最終資金 | 回収率 |
|:---:|:---:|:---:|
| Full Kelly (上限20%) | 826.82 | **144.18%** |
| Half Kelly (上限10%) | 446.43 | **165.54%** |
| Quarter Kelly (上限5%) | 247.21 | **179.52%** |

### 日本代表 スコア予測精度 (2022年)

| 試合 | 予測スコア | 実際 | 結果 |
|:---|:---:|:---:|:---:|
| ドイツ vs 日本 | 1-0 | 1-2 | 日本勝利🎉 |
| 日本 vs コスタリカ | 1-0 | 0-1 | 日本敗北 |
| 日本 vs スペイン | 0-1 | 2-1 | 日本勝利🎉 |
| 日本 vs クロアチア | **1-1** | **1-1** | **的中🎯** |

---

## 2026年大会 予測結果（2026年6月11日データ基準・大会開幕時点）

### 大会フォーマット
- **48チーム** / **12グループ×4チーム** / グループ上位2+ベスト3位8チームがR32進出
- **総試合数:** 104試合（グループステージ72 + ノックアウト32）
- **開催国:** アメリカ・カナダ・メキシコ（3カ国）

### 優勝確率シミュレーション TOP 12（1万回・公式ブラケット使用）

| 順位 | チーム | グループ | Elo | 決勝進出 | **優勝確率** |
|:---:|:---|:---:|:---:|:---:|:---:|
| 1 | 🇪🇸 スペイン | H | 2176 | 32.7% | **22.86%** |
| 2 | 🇫🇷 フランス | I | 2095 | 17.5% | **10.60%** |
| 3 | 🇦🇷 アルゼンチン | J | 2154 | 17.8% | **9.77%** |
| 4 | 🇧🇷 ブラジル | C | 2033 | 15.6% | **8.45%** |
| 5 | 🏴󠁧󠁢󠁥󠁮󠁧󠁿 イングランド | L | 2064 | 14.9% | **7.45%** |
| 6 | 🇨🇴 コロンビア | K | 2028 | 12.6% | **6.35%** |
| 7 | 🇲🇽 メキシコ | A | 1957 | 10.0% | **4.67%** |
| 8 | 🇩🇪 ドイツ | E | 1972 | 8.2% | **3.57%** |
| 9 | 🇵🇹 ポルトガル | K | 1993 | 8.0% | **3.52%** |
| 10 | 🇳🇱 オランダ | F | 1988 | 6.5% | **2.75%** |
| - | 🇯🇵 **日本** | **F** | 1964 | 3.9% | **1.50%** |
| - | 🇺🇸 アメリカ | D | 1813 | 3.0% | 0.99% |

> 開催国の `was_home` 適用と公式ブラケットの導入により、メキシコ（7位, 4.67%）など開催国の評価が上昇。

### 🇯🇵 グループF 日本代表 スコア予測

| 試合 | 予測スコア | 期待得点 | 日本勝率 |
|:---|:---:|:---:|:---:|
| **オランダ vs 日本** (6/14) | **1-1** | 1.46 - 1.13 | **27.6%** (引き分け28.6%) |
| **日本 vs チュニジア** (6/20) | **1-1** | 1.73 - 0.87 | **57.6%** |
| **日本 vs スウェーデン** (6/25) | **1-1** | 1.82 - 0.93 | **58.1%** |

> R32（グループ突破）確率: **85.3%** / R16進出確率: **41.3%**

### 2026年ベッティング推奨の例（全72試合 実オッズ・市場シュリンク0.5適用後）

| 試合 | 賭けタイプ | オッズ | **EV** |
|:---|:---|:---:|:---:|
| ブラジル vs ハイチ (6/13) | ハイチ勝利 | 28.89 | **1.819** |
| ノルウェー vs イラク (6/17) | DC X2 | 4.84 | **1.512** |
| 日本 vs スウェーデン (6/25) | 日本勝利 | 2.07 | **1.070** |

> EVが2.0を超える推奨はサニティチェックで自動除外される（オッズデータ異常の疑い）。
> 弱小国相手の高オッズには依然高EVが出やすい（モデルと市場の見解相違）。
> 実際の購入は「ベッティング戦略と資金管理ガイド」のEV閾値・ケリー基準に従うこと。

---

## データ更新フロー（大会中）

試合結果が蓄積されるたびに以下を実行することで、最新のEloとローリング統計に基づく予測に更新できます：

```bash
# 最新データ取得 → Elo再計算 → 特徴量更新 → 予測更新
python src/pipeline/download_data.py && python src/pipeline/elo.py && python src/pipeline/preprocess.py
python src/predict/predict_scores_2026.py   # スコア予測更新
python src/predict/simulator_2026.py        # 優勝確率更新
python src/odds/fetch_odds_2026.py          # 実オッズの最新化
python src/predict/bet_advisor_2026.py      # ベッティング推奨更新
python src/predict/settle_bets_2026.py      # WINNER購入結果の自動精算とROI確認

# （任意・推奨）まとまった試合数が消化されたらモデル自体も再学習
python src/pipeline/train_model.py
```

---

## ベッティング戦略と資金管理ガイド

本モデルを実際の予測やベッティング（WINNER、ブックメーカーなど）に適用する際は、以下の投資戦略と資金管理（マネーマネジメント）の原則に従うことを強く推奨します。

### 1. 期待値（EV）の判断基準と閾値

期待値 $EV = \sum (確率 \times オッズ)$ に基づいて購入を判断しますが、モデルの予測誤差や市場の急変に対応するため、以下の閾値を適用します。

*   **推奨購入ライン (EV ≧ 1.10 〜 1.15)**
    *   モデルの統計的ブレ（約5%〜10%）を考慮した**安全マージン（Margin of Safety）**です。期待利回りが10%〜15%以上の、明らかに「市場オッズが歪んでいる（過小評価されている）」選択肢に絞って購入します。
*   **見送りライン (EV < 1.05)**
    *   EVが1.00をわずかに超えているだけの場合、モデルのわずかな予測誤差で即座に期待値マイナスに転落するリスクがあるため、原則として見送ります。

### 2. WINNER（日本国内向けスポーツ振興くじ）特有の対策

WINNERはブックメーカーと異なり、非常に極端な市場バイアスが存在するため、以下の鉄則を守る必要があります。

1.  **「還元率50%」の壁を越える狙い撃ち**
    *   WINNERの還元率は50%（控除率50%）と非常に低いため、適当に購入しても資金は目減りします。EVが1.10を超えるような極端な歪みのみをターゲットにしてください。
2.  **応援バイアス（日本ひいき）の逆張り**
    *   日本国内限定のくじであるため、日本代表戦では「日本勝利」に購入が過剰集中し、日本勝利のオッズは著しく美味しくない水準に暴落します。
    *   期待値がプラスになり得るのは、逆の**「対戦相手の勝利」**や**「日本の引き分け（0-0など）」**に、バイアスによるオッズ高騰が発生した時のみです。感情を排除して数字に機械的に従うことがWINNER攻略の鍵です。

### 3. 資金管理とケリー基準（破産確率の低減）

いくらEVが高くても、1回あたりの購入金額が大きすぎたり、的中確率が低すぎるベットに全力を出すと、短期間の下振れで資金がショート（破産）します。

*   **ケリー基準の適用（ハーフケリーを推奨）**
    *   本システムでは、ケリー基準 $f^* = \frac{p \times b - q}{b}$ (但し、$b = オッズ-1$, $p = 確率$, $q = 1-p$) に基づき、最適なベット比率を計算します。
    *   安全のため、計算された比率の半分を賭ける**「ハーフケリー（Half-Kelly）基準」**（本スクリプトのデフォルト：最大10%上限）を採用し、資金の急激な下振れを抑制します。
*   **高オッズ・低確率の制限**
    *   オッズが15倍以上（的中確率10%未満）のような高オッズのスコア予想は、期待値が高くても的中が極めて稀です。同様のベットチャンスが限られるため、推奨ベット額をさらに半分（クォーターケリー）にするか、見送ることでドローダウンを抑制してください。