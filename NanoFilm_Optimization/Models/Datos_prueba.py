# -*- coding: utf-8 -*-
"""
Created on Sat Oct  7 13:28:59 2023

@author: 52331
"""


import pandas as pd
import itertools

# Definir el rango para las columnas A, B y C en metros (0.00000001 a 0.0000002 metros)
rango_metros = [valor * 1e-9 for valor in range(10, 201, 10)]

# Generar todas las combinaciones posibles de A, B y C
combinaciones = list(itertools.product(rango_metros, repeat=3))

# Crear un DataFrame con las combinaciones
df = pd.DataFrame(combinaciones, columns=['espTa2O5', 'espAl2O3', 'espSiO'])

# Establecer el valor fijo de la columna 'espTa2O5' en 0.0000001 metros (100 nm)
df['espTa2O5'] = 1e-7

# Valores de lambda que se agregarán a las filas
valores_lambda = [596.42147, 604.83871, 613.49693]

# Crear una nueva lista de filas que contendrá las filas duplicadas con diferentes valores de lambda
nuevas_filas = []

for index, row in df.iterrows():
    for valor_lambda in valores_lambda:
        nueva_fila = row.copy()
        nueva_fila['Lambda'] = valor_lambda
        nuevas_filas.append(nueva_fila)

# Crear un nuevo DataFrame con las filas duplicadas
df_nuevo = pd.DataFrame(nuevas_filas)