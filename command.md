# よく使うコマンド集（2026年大会 運用用）

実際の運用の時系列順に並んでいる。毎日のルーチンは 1 → 3 → 4、購入する日は加えて 5 → 6。

  ┌─────┬───────────────┬────────────────────────────────────────────────────────┐
  │  #  │  タイミング     │                          内容                          │
  ├─────┼───────────────┼────────────────────────────────────────────────────────┤
  │  0  │ 初回のみ        │ 環境・推定オッズ初期生成                                  │
  ├─────┼───────────────┼────────────────────────────────────────────────────────┤
  │  1  │ 毎朝           │ データ更新（download→Elo→特徴量）＋前日ベットの自動精        │
  │     │               │ 算                                                     │
  ├─────┼───────────────┼────────────────────────────────────────────────────────┤
  │  2  │ 情報があれば    │ 負傷者反映（penalties CSV）／数試合ごとの再学習              │
  ├─────┼───────────────┼────────────────────────────────────────────────────────┤
  │  3  │ 毎日           │ 予測・シミュレーション・オッズ取得・推奨（1+3のワンラ          │
  │     │               │ イナー付き）                                             │
  ├─────┼───────────────┼────────────────────────────────────────────────────────┤
  │  4  │ 購入する試合    │ WINNER 18択EV計算（全72試合コマンド）                      │
  │     │ のみ           │                                                        │
  ├─────┼───────────────┼────────────────────────────────────────────────────────┤
  │  5  │ 購入したら      │ buy_status.csv への記録（記入例付き）                     │
  ├─────┼───────────────┼────────────────────────────────────────────────────────┤
  │  6  │ 試合後         │ 結果精算（即時 or 翌朝自動）                               │
  ├─────┼───────────────┼────────────────────────────────────────────────────────┤
  │  7  │ 随時           │ バックテスト・ウォークフォワード検証                         │
  ├─────┼───────────────┼────────────────────────────────────────────────────────┤
  │ 8-9 │ リファレンス    │ 引数早見表・出力ファイル早見表                              │
  └─────┴───────────────┴────────────────────────────────────────────────────────┘

## 0. 環境（初回のみ）

```bash
conda activate wcup_winner
# もしくはフルパスで直接実行
# /opt/anaconda3/envs/wcup_winner/bin/python ...

# 推定オッズCSVの初期生成（初回のみ。実行済み）
# python src/odds/prepare_odds_2026_groups.py
```

---

## 1. 【毎朝】データ更新と前日ベットの自動精算

```bash
# 最新の試合結果を取得 → Elo再計算 → 特徴量再生成
python src/pipeline/download_data.py
python src/pipeline/elo.py
python src/pipeline/preprocess.py

# 前日までのWINNER購入を自動精算・通算ROI表示（結果が取り込まれた試合分）
python src/predict/settle_bets_2026.py
```

---

## 2. 【手作業】手動で拾ってくる情報（無料運用）

> 試合結果・日程・ブックメーカーオッズ・Elo・市場価値は自動取得済み。
> 手作業が必要なのは下記の4種類だけ。
> （API-Football有料プラン($19/月)に課金すれば (a)(b) は
>  `python src/pipeline/fetch_injuries_2026.py` / `--lineups` で自動化可能）

### (a) 負傷者・出場停止【購入予定チームのみ・購入前日まで】
- **拾う情報**: スタメン級選手の離脱（負傷・累積警告・出場停止）。控え選手は無視してよい
- **情報源**: FotMob / Flashscore / Transfermarktの負傷者ページ / 代表公式SNS
- **記入先**: `data/raw/squad/squad_penalties_2026.csv` に `チーム名,倍率,理由`
  - 目安: スタメン1人離脱 = 0.95、エース級 = 0.90、複数主力 = 0.85（下限の目安）
  - 例) `Japan,0.90,エースFW負傷離脱`
  - 復帰したら行を削除
- **反映**: セクション3の予測を再実行した時点から有効

### (b) スタメン発表【キックオフ約1時間前・購入直前の最終確認】
- **拾う情報**: 先発11人。特にグループ3戦目の「主力温存（5人以上入れ替え）」
- **使い方**: 大量ターンオーバーを確認したら、(a) の要領で一時的に倍率を下げて
  予測→WINNER EVを再計算してから購入判断（買わない判断も含む）

### (c) WINNERの18択オッズ【購入を検討する試合のみ】
- **拾う情報**: WINNERサイトの18択オッズ（スクリーンショットでOK）
- **記入先**: `data/raw/odds/winner_inputs/<連番>_winner_<home>_<away>.csv` の odds 列
  （セクション4のテンプレート生成→記入→計算）

### (d) 試合結果【すぐ精算したい場合のみ】
- **記入先**: `data/buy_status.csv` の `match_result` 列に「2-0」形式（購入行のhome/away視点）
- 急がなければ記入不要（翌朝の自動精算に任せる）

数試合消化ごと（2〜3日に1回程度）はモデル自体も再学習する:

```bash
# 本番モデル（全データ → models/ に保存。30分前後）
python src/pipeline/train_model.py
```

---

## 3. 【毎日】予測・ベッティング推奨の更新

```bash
# スコア予測（全72試合。1X2確率は Poisson行列×分類器のブレンド, --cls_blend で調整可）
python src/predict/predict_scores_2026.py

# 48カ国 優勝確率シミュレーション（1万回モンテカルロ・公式ブラケット）
python src/predict/simulator_2026.py

# 実オッズの最新化 (.env の THE_ODDS_API_KEY を使用)
python src/odds/fetch_odds_2026.py

# ベッティング推奨（EV・ケリー計算）
python src/predict/bet_advisor_2026.py --ev_thresh 1.05 --kelly half --bankroll 100
```

セクション1+3のワンライナー版:

```bash
python src/pipeline/download_data.py && python src/pipeline/elo.py && python src/pipeline/preprocess.py && \
python src/predict/settle_bets_2026.py && \
python src/predict/predict_scores_2026.py && python src/predict/simulator_2026.py && \
python src/odds/fetch_odds_2026.py && python src/predict/bet_advisor_2026.py
```

---

## 4. 【購入する試合のみ】WINNER 18択の期待値計算

手順: ①対象試合のテンプレート生成コマンドを実行 → ②生成されたCSVの `odds` 列に
WINNERサイトの18択オッズを記入 → ③同じ試合の計算コマンドを実行
（出力: `data/processed/2026/winner_matches/<連番>_winner_<home>_<away>_ev.csv`）

- λは `predicted_scores.csv` から読むため、**セクション3の predict_scores_2026.py を先に実行しておくこと**。
- モデル更新後の再計算は計算コマンドの再実行だけでよい
  （**テンプレートは作り直さない**。記入済みオッズが消えるため。01・02は記入済み）。

### グループステージ全72試合（日程順）

```bash
# ===== 6/11 =====
# 01: Mexico vs South Africa
python src/predict/calculate_winner_ev_general.py --generate_template data/raw/odds/winner_inputs/01_winner_mexico_south_africa.csv
python src/predict/calculate_winner_ev_general.py --home Mexico --away "South Africa" --odds_csv data/raw/odds/winner_inputs/01_winner_mexico_south_africa.csv
# 02: South Korea vs Czech Republic
python src/predict/calculate_winner_ev_general.py --generate_template data/raw/odds/winner_inputs/02_winner_south_korea_czech_republic.csv
python src/predict/calculate_winner_ev_general.py --home "South Korea" --away "Czech Republic" --odds_csv data/raw/odds/winner_inputs/02_winner_south_korea_czech_republic.csv

# ===== 6/12 =====
# 03: Canada vs Bosnia and Herzegovina
python src/predict/calculate_winner_ev_general.py --generate_template data/raw/odds/winner_inputs/03_winner_canada_bosnia_and_herzegovina.csv
python src/predict/calculate_winner_ev_general.py --home Canada --away "Bosnia and Herzegovina" --odds_csv data/raw/odds/winner_inputs/03_winner_canada_bosnia_and_herzegovina.csv
# 04: United States vs Paraguay
python src/predict/calculate_winner_ev_general.py --generate_template data/raw/odds/winner_inputs/04_winner_united_states_paraguay.csv
python src/predict/calculate_winner_ev_general.py --home "United States" --away Paraguay --odds_csv data/raw/odds/winner_inputs/04_winner_united_states_paraguay.csv

# ===== 6/13 =====
# 05: Qatar vs Switzerland
python src/predict/calculate_winner_ev_general.py --generate_template data/raw/odds/winner_inputs/05_winner_qatar_switzerland.csv
python src/predict/calculate_winner_ev_general.py --home Qatar --away Switzerland --odds_csv data/raw/odds/winner_inputs/05_winner_qatar_switzerland.csv
# 06: Brazil vs Morocco
python src/predict/calculate_winner_ev_general.py --generate_template data/raw/odds/winner_inputs/06_winner_brazil_morocco.csv
python src/predict/calculate_winner_ev_general.py --home Brazil --away Morocco --odds_csv data/raw/odds/winner_inputs/06_winner_brazil_morocco.csv
# 07: Haiti vs Scotland
python src/predict/calculate_winner_ev_general.py --generate_template data/raw/odds/winner_inputs/07_winner_haiti_scotland.csv
python src/predict/calculate_winner_ev_general.py --home Haiti --away Scotland --odds_csv data/raw/odds/winner_inputs/07_winner_haiti_scotland.csv
# 08: Australia vs Turkey
python src/predict/calculate_winner_ev_general.py --generate_template data/raw/odds/winner_inputs/08_winner_australia_turkey.csv
python src/predict/calculate_winner_ev_general.py --home Australia --away Turkey --odds_csv data/raw/odds/winner_inputs/08_winner_australia_turkey.csv

# ===== 6/14 =====
# 09: Germany vs Curaçao
python src/predict/calculate_winner_ev_general.py --generate_template data/raw/odds/winner_inputs/09_winner_germany_curaçao.csv
python src/predict/calculate_winner_ev_general.py --home Germany --away Curaçao --odds_csv data/raw/odds/winner_inputs/09_winner_germany_curaçao.csv
# 10: Ivory Coast vs Ecuador
python src/predict/calculate_winner_ev_general.py --generate_template data/raw/odds/winner_inputs/10_winner_ivory_coast_ecuador.csv
python src/predict/calculate_winner_ev_general.py --home "Ivory Coast" --away Ecuador --odds_csv data/raw/odds/winner_inputs/10_winner_ivory_coast_ecuador.csv
# 11: Netherlands vs Japan
python src/predict/calculate_winner_ev_general.py --generate_template data/raw/odds/winner_inputs/11_winner_netherlands_japan.csv
python src/predict/calculate_winner_ev_general.py --home Netherlands --away Japan --odds_csv data/raw/odds/winner_inputs/11_winner_netherlands_japan.csv
# 12: Sweden vs Tunisia
python src/predict/calculate_winner_ev_general.py --generate_template data/raw/odds/winner_inputs/12_winner_sweden_tunisia.csv
python src/predict/calculate_winner_ev_general.py --home Sweden --away Tunisia --odds_csv data/raw/odds/winner_inputs/12_winner_sweden_tunisia.csv

# ===== 6/15 =====
# 13: Belgium vs Egypt
python src/predict/calculate_winner_ev_general.py --generate_template data/raw/odds/winner_inputs/13_winner_belgium_egypt.csv
python src/predict/calculate_winner_ev_general.py --home Belgium --away Egypt --odds_csv data/raw/odds/winner_inputs/13_winner_belgium_egypt.csv
# 14: Iran vs New Zealand
python src/predict/calculate_winner_ev_general.py --generate_template data/raw/odds/winner_inputs/14_winner_iran_new_zealand.csv
python src/predict/calculate_winner_ev_general.py --home Iran --away "New Zealand" --odds_csv data/raw/odds/winner_inputs/14_winner_iran_new_zealand.csv
# 15: Spain vs Cape Verde
python src/predict/calculate_winner_ev_general.py --generate_template data/raw/odds/winner_inputs/15_winner_spain_cape_verde.csv
python src/predict/calculate_winner_ev_general.py --home Spain --away "Cape Verde" --odds_csv data/raw/odds/winner_inputs/15_winner_spain_cape_verde.csv
# 16: Saudi Arabia vs Uruguay
python src/predict/calculate_winner_ev_general.py --generate_template data/raw/odds/winner_inputs/16_winner_saudi_arabia_uruguay.csv
python src/predict/calculate_winner_ev_general.py --home "Saudi Arabia" --away Uruguay --odds_csv data/raw/odds/winner_inputs/16_winner_saudi_arabia_uruguay.csv

# ===== 6/16 =====
# 17: France vs Senegal
python src/predict/calculate_winner_ev_general.py --generate_template data/raw/odds/winner_inputs/17_winner_france_senegal.csv
python src/predict/calculate_winner_ev_general.py --home France --away Senegal --odds_csv data/raw/odds/winner_inputs/17_winner_france_senegal.csv
# 18: Iraq vs Norway
python src/predict/calculate_winner_ev_general.py --generate_template data/raw/odds/winner_inputs/18_winner_iraq_norway.csv
python src/predict/calculate_winner_ev_general.py --home Iraq --away Norway --odds_csv data/raw/odds/winner_inputs/18_winner_iraq_norway.csv
# 19: Argentina vs Algeria
python src/predict/calculate_winner_ev_general.py --generate_template data/raw/odds/winner_inputs/19_winner_argentina_algeria.csv
python src/predict/calculate_winner_ev_general.py --home Argentina --away Algeria --odds_csv data/raw/odds/winner_inputs/19_winner_argentina_algeria.csv
# 20: Austria vs Jordan
python src/predict/calculate_winner_ev_general.py --generate_template data/raw/odds/winner_inputs/20_winner_austria_jordan.csv
python src/predict/calculate_winner_ev_general.py --home Austria --away Jordan --odds_csv data/raw/odds/winner_inputs/20_winner_austria_jordan.csv

# ===== 6/17 =====
# 21: Portugal vs DR Congo
python src/predict/calculate_winner_ev_general.py --generate_template data/raw/odds/winner_inputs/21_winner_portugal_dr_congo.csv
python src/predict/calculate_winner_ev_general.py --home Portugal --away "DR Congo" --odds_csv data/raw/odds/winner_inputs/21_winner_portugal_dr_congo.csv
# 22: Uzbekistan vs Colombia
python src/predict/calculate_winner_ev_general.py --generate_template data/raw/odds/winner_inputs/22_winner_uzbekistan_colombia.csv
python src/predict/calculate_winner_ev_general.py --home Uzbekistan --away Colombia --odds_csv data/raw/odds/winner_inputs/22_winner_uzbekistan_colombia.csv
# 23: England vs Croatia
python src/predict/calculate_winner_ev_general.py --generate_template data/raw/odds/winner_inputs/23_winner_england_croatia.csv
python src/predict/calculate_winner_ev_general.py --home England --away Croatia --odds_csv data/raw/odds/winner_inputs/23_winner_england_croatia.csv
# 24: Ghana vs Panama
python src/predict/calculate_winner_ev_general.py --generate_template data/raw/odds/winner_inputs/24_winner_ghana_panama.csv
python src/predict/calculate_winner_ev_general.py --home Ghana --away Panama --odds_csv data/raw/odds/winner_inputs/24_winner_ghana_panama.csv

# ===== 6/18 =====
# 25: Czech Republic vs South Africa
python src/predict/calculate_winner_ev_general.py --generate_template data/raw/odds/winner_inputs/25_winner_czech_republic_south_africa.csv
python src/predict/calculate_winner_ev_general.py --home "Czech Republic" --away "South Africa" --odds_csv data/raw/odds/winner_inputs/25_winner_czech_republic_south_africa.csv
# 26: Mexico vs South Korea
python src/predict/calculate_winner_ev_general.py --generate_template data/raw/odds/winner_inputs/26_winner_mexico_south_korea.csv
python src/predict/calculate_winner_ev_general.py --home Mexico --away "South Korea" --odds_csv data/raw/odds/winner_inputs/26_winner_mexico_south_korea.csv
# 27: Switzerland vs Bosnia and Herzegovina
python src/predict/calculate_winner_ev_general.py --generate_template data/raw/odds/winner_inputs/27_winner_switzerland_bosnia_and_herzegovina.csv
python src/predict/calculate_winner_ev_general.py --home Switzerland --away "Bosnia and Herzegovina" --odds_csv data/raw/odds/winner_inputs/27_winner_switzerland_bosnia_and_herzegovina.csv
# 28: Canada vs Qatar
python src/predict/calculate_winner_ev_general.py --generate_template data/raw/odds/winner_inputs/28_winner_canada_qatar.csv
python src/predict/calculate_winner_ev_general.py --home Canada --away Qatar --odds_csv data/raw/odds/winner_inputs/28_winner_canada_qatar.csv

# ===== 6/19 =====
# 29: Scotland vs Morocco
python src/predict/calculate_winner_ev_general.py --generate_template data/raw/odds/winner_inputs/29_winner_scotland_morocco.csv
python src/predict/calculate_winner_ev_general.py --home Scotland --away Morocco --odds_csv data/raw/odds/winner_inputs/29_winner_scotland_morocco.csv
# 30: Brazil vs Haiti
python src/predict/calculate_winner_ev_general.py --generate_template data/raw/odds/winner_inputs/30_winner_brazil_haiti.csv
python src/predict/calculate_winner_ev_general.py --home Brazil --away Haiti --odds_csv data/raw/odds/winner_inputs/30_winner_brazil_haiti.csv
# 31: United States vs Australia
python src/predict/calculate_winner_ev_general.py --generate_template data/raw/odds/winner_inputs/31_winner_united_states_australia.csv
python src/predict/calculate_winner_ev_general.py --home "United States" --away Australia --odds_csv data/raw/odds/winner_inputs/31_winner_united_states_australia.csv
# 32: Turkey vs Paraguay
python src/predict/calculate_winner_ev_general.py --generate_template data/raw/odds/winner_inputs/32_winner_turkey_paraguay.csv
python src/predict/calculate_winner_ev_general.py --home Turkey --away Paraguay --odds_csv data/raw/odds/winner_inputs/32_winner_turkey_paraguay.csv

# ===== 6/20 =====
# 33: Germany vs Ivory Coast
python src/predict/calculate_winner_ev_general.py --generate_template data/raw/odds/winner_inputs/33_winner_germany_ivory_coast.csv
python src/predict/calculate_winner_ev_general.py --home Germany --away "Ivory Coast" --odds_csv data/raw/odds/winner_inputs/33_winner_germany_ivory_coast.csv
# 34: Ecuador vs Curaçao
python src/predict/calculate_winner_ev_general.py --generate_template data/raw/odds/winner_inputs/34_winner_ecuador_curaçao.csv
python src/predict/calculate_winner_ev_general.py --home Ecuador --away Curaçao --odds_csv data/raw/odds/winner_inputs/34_winner_ecuador_curaçao.csv
# 35: Netherlands vs Sweden
python src/predict/calculate_winner_ev_general.py --generate_template data/raw/odds/winner_inputs/35_winner_netherlands_sweden.csv
python src/predict/calculate_winner_ev_general.py --home Netherlands --away Sweden --odds_csv data/raw/odds/winner_inputs/35_winner_netherlands_sweden.csv
# 36: Tunisia vs Japan
python src/predict/calculate_winner_ev_general.py --generate_template data/raw/odds/winner_inputs/36_winner_tunisia_japan.csv
python src/predict/calculate_winner_ev_general.py --home Tunisia --away Japan --odds_csv data/raw/odds/winner_inputs/36_winner_tunisia_japan.csv

# ===== 6/21 =====
# 37: Belgium vs Iran
python src/predict/calculate_winner_ev_general.py --generate_template data/raw/odds/winner_inputs/37_winner_belgium_iran.csv
python src/predict/calculate_winner_ev_general.py --home Belgium --away Iran --odds_csv data/raw/odds/winner_inputs/37_winner_belgium_iran.csv
# 38: New Zealand vs Egypt
python src/predict/calculate_winner_ev_general.py --generate_template data/raw/odds/winner_inputs/38_winner_new_zealand_egypt.csv
python src/predict/calculate_winner_ev_general.py --home "New Zealand" --away Egypt --odds_csv data/raw/odds/winner_inputs/38_winner_new_zealand_egypt.csv
# 39: Spain vs Saudi Arabia
python src/predict/calculate_winner_ev_general.py --generate_template data/raw/odds/winner_inputs/39_winner_spain_saudi_arabia.csv
python src/predict/calculate_winner_ev_general.py --home Spain --away "Saudi Arabia" --odds_csv data/raw/odds/winner_inputs/39_winner_spain_saudi_arabia.csv
# 40: Uruguay vs Cape Verde
python src/predict/calculate_winner_ev_general.py --generate_template data/raw/odds/winner_inputs/40_winner_uruguay_cape_verde.csv
python src/predict/calculate_winner_ev_general.py --home Uruguay --away "Cape Verde" --odds_csv data/raw/odds/winner_inputs/40_winner_uruguay_cape_verde.csv

# ===== 6/22 =====
# 41: France vs Iraq
python src/predict/calculate_winner_ev_general.py --generate_template data/raw/odds/winner_inputs/41_winner_france_iraq.csv
python src/predict/calculate_winner_ev_general.py --home France --away Iraq --odds_csv data/raw/odds/winner_inputs/41_winner_france_iraq.csv
# 42: Norway vs Senegal
python src/predict/calculate_winner_ev_general.py --generate_template data/raw/odds/winner_inputs/42_winner_norway_senegal.csv
python src/predict/calculate_winner_ev_general.py --home Norway --away Senegal --odds_csv data/raw/odds/winner_inputs/42_winner_norway_senegal.csv
# 43: Argentina vs Austria
python src/predict/calculate_winner_ev_general.py --generate_template data/raw/odds/winner_inputs/43_winner_argentina_austria.csv
python src/predict/calculate_winner_ev_general.py --home Argentina --away Austria --odds_csv data/raw/odds/winner_inputs/43_winner_argentina_austria.csv
# 44: Jordan vs Algeria
python src/predict/calculate_winner_ev_general.py --generate_template data/raw/odds/winner_inputs/44_winner_jordan_algeria.csv
python src/predict/calculate_winner_ev_general.py --home Jordan --away Algeria --odds_csv data/raw/odds/winner_inputs/44_winner_jordan_algeria.csv

# ===== 6/23 =====
# 45: Portugal vs Uzbekistan
python src/predict/calculate_winner_ev_general.py --generate_template data/raw/odds/winner_inputs/45_winner_portugal_uzbekistan.csv
python src/predict/calculate_winner_ev_general.py --home Portugal --away Uzbekistan --odds_csv data/raw/odds/winner_inputs/45_winner_portugal_uzbekistan.csv
# 46: Colombia vs DR Congo
python src/predict/calculate_winner_ev_general.py --generate_template data/raw/odds/winner_inputs/46_winner_colombia_dr_congo.csv
python src/predict/calculate_winner_ev_general.py --home Colombia --away "DR Congo" --odds_csv data/raw/odds/winner_inputs/46_winner_colombia_dr_congo.csv
# 47: England vs Ghana
python src/predict/calculate_winner_ev_general.py --generate_template data/raw/odds/winner_inputs/47_winner_england_ghana.csv
python src/predict/calculate_winner_ev_general.py --home England --away Ghana --odds_csv data/raw/odds/winner_inputs/47_winner_england_ghana.csv
# 48: Panama vs Croatia
python src/predict/calculate_winner_ev_general.py --generate_template data/raw/odds/winner_inputs/48_winner_panama_croatia.csv
python src/predict/calculate_winner_ev_general.py --home Panama --away Croatia --odds_csv data/raw/odds/winner_inputs/48_winner_panama_croatia.csv

# ===== 6/24 =====
# 49: Mexico vs Czech Republic
python src/predict/calculate_winner_ev_general.py --generate_template data/raw/odds/winner_inputs/49_winner_mexico_czech_republic.csv
python src/predict/calculate_winner_ev_general.py --home Mexico --away "Czech Republic" --odds_csv data/raw/odds/winner_inputs/49_winner_mexico_czech_republic.csv
# 50: South Africa vs South Korea
python src/predict/calculate_winner_ev_general.py --generate_template data/raw/odds/winner_inputs/50_winner_south_africa_south_korea.csv
python src/predict/calculate_winner_ev_general.py --home "South Africa" --away "South Korea" --odds_csv data/raw/odds/winner_inputs/50_winner_south_africa_south_korea.csv
# 51: Canada vs Switzerland
python src/predict/calculate_winner_ev_general.py --generate_template data/raw/odds/winner_inputs/51_winner_canada_switzerland.csv
python src/predict/calculate_winner_ev_general.py --home Canada --away Switzerland --odds_csv data/raw/odds/winner_inputs/51_winner_canada_switzerland.csv
# 52: Bosnia and Herzegovina vs Qatar
python src/predict/calculate_winner_ev_general.py --generate_template data/raw/odds/winner_inputs/52_winner_bosnia_and_herzegovina_qatar.csv
python src/predict/calculate_winner_ev_general.py --home "Bosnia and Herzegovina" --away Qatar --odds_csv data/raw/odds/winner_inputs/52_winner_bosnia_and_herzegovina_qatar.csv
# 53: Scotland vs Brazil
python src/predict/calculate_winner_ev_general.py --generate_template data/raw/odds/winner_inputs/53_winner_scotland_brazil.csv
python src/predict/calculate_winner_ev_general.py --home Scotland --away Brazil --odds_csv data/raw/odds/winner_inputs/53_winner_scotland_brazil.csv
# 54: Morocco vs Haiti
python src/predict/calculate_winner_ev_general.py --generate_template data/raw/odds/winner_inputs/54_winner_morocco_haiti.csv
python src/predict/calculate_winner_ev_general.py --home Morocco --away Haiti --odds_csv data/raw/odds/winner_inputs/54_winner_morocco_haiti.csv

# ===== 6/25 =====
# 55: United States vs Turkey
python src/predict/calculate_winner_ev_general.py --generate_template data/raw/odds/winner_inputs/55_winner_united_states_turkey.csv
python src/predict/calculate_winner_ev_general.py --home "United States" --away Turkey --odds_csv data/raw/odds/winner_inputs/55_winner_united_states_turkey.csv
# 56: Paraguay vs Australia
python src/predict/calculate_winner_ev_general.py --generate_template data/raw/odds/winner_inputs/56_winner_paraguay_australia.csv
python src/predict/calculate_winner_ev_general.py --home Paraguay --away Australia --odds_csv data/raw/odds/winner_inputs/56_winner_paraguay_australia.csv
# 57: Curaçao vs Ivory Coast
python src/predict/calculate_winner_ev_general.py --generate_template data/raw/odds/winner_inputs/57_winner_curaçao_ivory_coast.csv
python src/predict/calculate_winner_ev_general.py --home Curaçao --away "Ivory Coast" --odds_csv data/raw/odds/winner_inputs/57_winner_curaçao_ivory_coast.csv
# 58: Ecuador vs Germany
python src/predict/calculate_winner_ev_general.py --generate_template data/raw/odds/winner_inputs/58_winner_ecuador_germany.csv
python src/predict/calculate_winner_ev_general.py --home Ecuador --away Germany --odds_csv data/raw/odds/winner_inputs/58_winner_ecuador_germany.csv
# 59: Japan vs Sweden
python src/predict/calculate_winner_ev_general.py --generate_template data/raw/odds/winner_inputs/59_winner_japan_sweden.csv
python src/predict/calculate_winner_ev_general.py --home Japan --away Sweden --odds_csv data/raw/odds/winner_inputs/59_winner_japan_sweden.csv
# 60: Tunisia vs Netherlands
python src/predict/calculate_winner_ev_general.py --generate_template data/raw/odds/winner_inputs/60_winner_tunisia_netherlands.csv
python src/predict/calculate_winner_ev_general.py --home Tunisia --away Netherlands --odds_csv data/raw/odds/winner_inputs/60_winner_tunisia_netherlands.csv

# ===== 6/26 =====
# 61: Egypt vs Iran
python src/predict/calculate_winner_ev_general.py --generate_template data/raw/odds/winner_inputs/61_winner_egypt_iran.csv
python src/predict/calculate_winner_ev_general.py --home Egypt --away Iran --odds_csv data/raw/odds/winner_inputs/61_winner_egypt_iran.csv
# 62: New Zealand vs Belgium
python src/predict/calculate_winner_ev_general.py --generate_template data/raw/odds/winner_inputs/62_winner_new_zealand_belgium.csv
python src/predict/calculate_winner_ev_general.py --home "New Zealand" --away Belgium --odds_csv data/raw/odds/winner_inputs/62_winner_new_zealand_belgium.csv
# 63: Cape Verde vs Saudi Arabia
python src/predict/calculate_winner_ev_general.py --generate_template data/raw/odds/winner_inputs/63_winner_cape_verde_saudi_arabia.csv
python src/predict/calculate_winner_ev_general.py --home "Cape Verde" --away "Saudi Arabia" --odds_csv data/raw/odds/winner_inputs/63_winner_cape_verde_saudi_arabia.csv
# 64: Uruguay vs Spain
python src/predict/calculate_winner_ev_general.py --generate_template data/raw/odds/winner_inputs/64_winner_uruguay_spain.csv
python src/predict/calculate_winner_ev_general.py --home Uruguay --away Spain --odds_csv data/raw/odds/winner_inputs/64_winner_uruguay_spain.csv
# 65: Norway vs France
python src/predict/calculate_winner_ev_general.py --generate_template data/raw/odds/winner_inputs/65_winner_norway_france.csv
python src/predict/calculate_winner_ev_general.py --home Norway --away France --odds_csv data/raw/odds/winner_inputs/65_winner_norway_france.csv
# 66: Senegal vs Iraq
python src/predict/calculate_winner_ev_general.py --generate_template data/raw/odds/winner_inputs/66_winner_senegal_iraq.csv
python src/predict/calculate_winner_ev_general.py --home Senegal --away Iraq --odds_csv data/raw/odds/winner_inputs/66_winner_senegal_iraq.csv

# ===== 6/27 =====
# 67: Algeria vs Austria
python src/predict/calculate_winner_ev_general.py --generate_template data/raw/odds/winner_inputs/67_winner_algeria_austria.csv
python src/predict/calculate_winner_ev_general.py --home Algeria --away Austria --odds_csv data/raw/odds/winner_inputs/67_winner_algeria_austria.csv
# 68: Jordan vs Argentina
python src/predict/calculate_winner_ev_general.py --generate_template data/raw/odds/winner_inputs/68_winner_jordan_argentina.csv
python src/predict/calculate_winner_ev_general.py --home Jordan --away Argentina --odds_csv data/raw/odds/winner_inputs/68_winner_jordan_argentina.csv
# 69: Colombia vs Portugal
python src/predict/calculate_winner_ev_general.py --generate_template data/raw/odds/winner_inputs/69_winner_colombia_portugal.csv
python src/predict/calculate_winner_ev_general.py --home Colombia --away Portugal --odds_csv data/raw/odds/winner_inputs/69_winner_colombia_portugal.csv
# 70: DR Congo vs Uzbekistan
python src/predict/calculate_winner_ev_general.py --generate_template data/raw/odds/winner_inputs/70_winner_dr_congo_uzbekistan.csv
python src/predict/calculate_winner_ev_general.py --home "DR Congo" --away Uzbekistan --odds_csv data/raw/odds/winner_inputs/70_winner_dr_congo_uzbekistan.csv
# 71: Panama vs England
python src/predict/calculate_winner_ev_general.py --generate_template data/raw/odds/winner_inputs/71_winner_panama_england.csv
python src/predict/calculate_winner_ev_general.py --home Panama --away England --odds_csv data/raw/odds/winner_inputs/71_winner_panama_england.csv
# 72: Croatia vs Ghana
python src/predict/calculate_winner_ev_general.py --generate_template data/raw/odds/winner_inputs/72_winner_croatia_ghana.csv
python src/predict/calculate_winner_ev_general.py --home Croatia --away Ghana --odds_csv data/raw/odds/winner_inputs/72_winner_croatia_ghana.csv
```

---

## 5. 【購入したら】購入の記録

```bash
# data/buy_status.csv に1行追記する（status=pending）
#   date,home_team,away_team,selection,odds,Probability(%),buy_amount_yen,match_result,result_amount_yen,status
#   例) 2026-06-14,Netherlands,Japan,1-1,7.5,12.3,600,,,pending
```

---

## 6. 【試合後】結果の精算

```bash
# すぐ精算したい場合: buy_status.csv の match_result 列に「2-0」のように記入
# （購入行のhome/away視点のスコア）してから実行
python src/predict/settle_bets_2026.py

# 急がない場合は何もしなくてよい（翌朝のセクション1で自動精算される）
```

---

## 7. 【随時】バックテスト・検証

```bash
# バックテスト用カットオフモデルの学習（作成済み。作り直す場合のみ）
python src/pipeline/train_model.py --train_end 2018-06-14 --model_dir models/backtest_2018
python src/pipeline/train_model.py --train_end 2022-11-20 --model_dir models/backtest_2022

# 大会バックテスト（models/backtest_{year}/ を自動使用 = リークなし）
python src/backtest/backtest.py --year 2022
python src/backtest/backtest.py --year 2018

# ウォークフォワード検証（パラメータ調整。全国際試合で評価。新特徴量・新モデルはここでLog Loss改善を確認してから本番投入）
python src/backtest/walkforward.py --mode probs --start_year 2018  # 成分予測の生成 + ρ×ブレンド比
python src/backtest/walkforward.py --mode ensemble # λアンサンブル構成の比較（GLM/XGB/チームID入りLGBM等）
python src/backtest/walkforward.py --mode matrix   # スコア行列: Dixon-Coles vs 二変量Poisson（スコアLLも評価）
python src/backtest/walkforward.py --mode stack    # スタッキング vs 固定ブレンド
python src/backtest/walkforward.py --mode shrink --rho -0.09 --cls_blend 0.25  # 市場シュリンク比
python src/backtest/walkforward.py --mode elo      # Elo Kスケール × ホーム補正

# WINNER市場のバイアス分析（蓄積したオッズ×モデル確率×実績をカテゴリ別に集計）
python src/predict/analyze_winner_bias.py

# xG検証（StatsBombの過去W杯データ取得 → xGの予測力分析。将来の特徴量化の判断材料）
python src/pipeline/fetch_statsbomb_xg.py
python src/backtest/analyze_xg_value.py
```

---

## 8. 主な引数（デフォルトは調整済みの最適値）

| スクリプト | 引数 | デフォルト | 意味 |
|:---|:---|:---:|:---|
| bet_advisor_2026.py | `--ev_thresh` | 1.05 | EV閾値（実購入は1.10〜1.15推奨） |
| | `--kelly` | half | full / half / quarter |
| | `--market_blend` | 0.5 | 市場確率へのシュリンク比 |
| | `--max_ev` | 2.0 | EVサニティ上限（超過は異常として除外） |
| predict_scores_2026.py | `--cls_blend` | 0.25 | 分類器ブレンド比 |
| train_model.py | `--train_end` | なし(全データ) | 学習終了日 |
| | `--model_dir` | models | モデル保存先 |
| | `--n_trials` | 30 | Optunaトライアル数 |
| backtest.py | `--year` | 2022 | 対象大会 |
| | `--model_dir` | 自動 | models/backtest_{year} 優先 |

---

## 9. 主な出力ファイル

| ファイル | 内容 |
|:---|:---|
| `data/processed/2026/predicted_scores.csv` | 全72試合のスコア予測・勝敗確率 |
| `data/processed/2026/simulation_results.csv` | 優勝確率（1万回モンテカルロ） |
| `data/processed/2026/bet_recommendations.csv` | EV・ケリー基準の推奨ベット |
| `data/processed/2026/winner_matches/*_ev.csv` | WINNER 18択の期待値 |
| `data/buy_status.csv` | WINNER購入履歴・収支管理 |
