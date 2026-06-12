#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
altitude.py - 開催都市の標高テーブルと標高差特徴量のルックアップ

標高はサッカーの国際試合で実証されている環境要因（高地の薄い空気に未順応の
チームはパフォーマンスが落ちる。CONMEBOL予選の La Paz / Quito / Bogotá が典型）。

特徴量の定義:
  altitude_diff = 開催都市の標高 - そのチームの本拠地（ホームスタジアム）の標高
  （正の値が大きいほど「普段より高い場所」での試合 = 不利になりやすい）

テーブルは高地開催が多い都市（概ね800m以上）と2026年大会の全16会場をカバーする。
未登録都市はデフォルト100m（平地）として扱う。
"""

DEFAULT_ALTITUDE = 100.0

# 開催都市 → 標高(m)。results.csv の city 列の表記に合わせる。
CITY_ALTITUDES = {
    # --- 南米の高地 ---
    'La Paz': 3640, 'El Alto': 4090, 'Potosí': 4070, 'Oruro': 3710,
    'Sucre': 2810, 'Cochabamba': 2560, 'Quito': 2850, 'Cuenca': 2550,
    'Ambato': 2580, 'Riobamba': 2750, 'Latacunga': 2760,
    'Bogotá': 2640, 'Pasto': 2530, 'Medellín': 1495, 'Cali': 1000,
    'Cusco': 3400, 'Huancayo': 3260, 'Juliaca': 3825, 'Cajamarca': 2750,
    'Arequipa': 2335, 'Brasília': 1170, 'Belo Horizonte': 850, 'São Paulo': 760,
    'Santiago': 570,
    # --- メキシコ・中米 ---
    'Mexico City': 2240, 'Toluca': 2660, 'Pachuca': 2430, 'Puebla': 2135,
    'León': 1815, 'Aguascalientes': 1880, 'Querétaro': 1820,
    'San Luis Potosí': 1860, 'Guadalajara': 1566, 'Zapopan': 1566,
    'Torreón': 1120, 'Monterrey': 540, 'Guadalupe': 500,
    'Guatemala City': 1500, 'Tegucigalpa': 990, 'San José': 1170,
    # --- 北米 ---
    'Denver': 1609, 'Commerce City': 1580, 'Salt Lake City': 1290,
    'Sandy': 1330, 'Calgary': 1045, 'Edmonton': 645,
    # --- アフリカの高地 ---
    'Addis Ababa': 2355, 'Asmara': 2325, 'Nairobi': 1795, 'Kampala': 1190,
    'Kigali': 1567, 'Harare': 1490, 'Bulawayo': 1350,
    'Johannesburg': 1753, 'Pretoria': 1339, 'Bloemfontein': 1395,
    'Rustenburg': 1500, 'Polokwane': 1310, 'Windhoek': 1655,
    'Lusaka': 1280, 'Ndola': 1270, 'Lilongwe': 1050, 'Blantyre': 1040,
    'Gaborone': 1010, 'Maseru': 1600, 'Mbabane': 1240,
    'Antananarivo': 1280, 'Bujumbura': 820,
    # --- アジア・欧州の高地 ---
    'Tehran': 1200, "Sana'a": 2250, 'Kabul': 1790, 'Kathmandu': 1400,
    'Thimphu': 2330, 'Quetta': 1680, 'Yerevan': 990, 'Ankara': 938,
    'Madrid': 667, 'Bern': 540, 'Munich': 519,
    # --- 2026年W杯 全16会場 ---
    # (Mexico City / Zapopan / Guadalupe は上で定義済み)
    'Arlington': 184, 'Atlanta': 320, 'East Rutherford': 3, 'Foxborough': 89,
    'Houston': 15, 'Inglewood': 37, 'Kansas City': 270, 'Miami Gardens': 3,
    'Philadelphia': 12, 'Santa Clara': 25, 'Seattle': 53,
    'Toronto': 76, 'Vancouver': 20,
}

# 国 → 代表チームの主たるホームスタジアムの標高(m)。未登録はデフォルト100m。
COUNTRY_HOME_ALTITUDES = {
    'Bolivia': 3640, 'Ecuador': 2850, 'Colombia': 2640, 'Mexico': 2240,
    'Ethiopia': 2355, 'Eritrea': 2325, 'Yemen': 2250, 'Bhutan': 2330,
    'Afghanistan': 1790, 'Nepal': 1400, 'Kenya': 1795, 'Rwanda': 1567,
    'Uganda': 1190, 'Zimbabwe': 1490, 'South Africa': 1700, 'Namibia': 1655,
    'Zambia': 1280, 'Malawi': 1050, 'Botswana': 1010, 'Lesotho': 1600,
    'Eswatini': 1240, 'Swaziland': 1240, 'Madagascar': 1280, 'Burundi': 820,
    'Iran': 1200, 'Armenia': 990, 'Guatemala': 1500, 'Honduras': 990,
    'Costa Rica': 1170, 'Switzerland': 400, 'Spain': 600,
}


def city_altitude(city):
    return CITY_ALTITUDES.get(str(city), DEFAULT_ALTITUDE)


def team_home_altitude(team):
    return COUNTRY_HOME_ALTITUDES.get(str(team), DEFAULT_ALTITUDE)


def altitude_diff(city, team):
    """開催都市の標高とチームの本拠標高の差（m）"""
    return city_altitude(city) - team_home_altitude(team)
