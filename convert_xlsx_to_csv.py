import pandas as pd
import sys

SRC = 'Viviendas.xlsx'
DST = 'Viviendas_limpio.csv'

try:
    df = pd.read_excel(SRC)
    df.to_csv(DST, index=False)
    print(f'Se creó {DST} con {len(df)} filas y {len(df.columns)} columnas.')
except Exception as e:
    print('ERROR al convertir:', e)
    sys.exit(1)
