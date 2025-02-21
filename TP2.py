print('################### Exercice 1 ###################')

# Importation de la bibliothèque pandas
import pandas as pd

# Création de la série Pandas à partir du dictionnaire
data = {'a': 100, 'b': 200, 'c': 300}
series = pd.Series(data)

# Affichage de la série
# print(series)
print('################### Exercice 2 ###################')
import pandas as pd

# Création du DataFrame
df = pd.DataFrame({
    'A': [1, 2, 3],
    'B': [4, 5, 6],
    'C': [7, 8, 9]
})

# Ajout de la colonne D
df['D'] = [10, 11, 12]

# Suppression de la colonne B
df = df.drop(columns=['B'])

print(df)

print('################### Exercice 3 ###################')

import pandas as pd

# Création du DataFrame
df = pd.DataFrame({
    'A': [1, 2, 3],
    'B': [4, 5, 6],
    'C': [7, 8, 9]
})

# Sélection de la colonne B
print(df['B'])

# Sélection des colonnes A et C
print(df[['A', 'C']])

# Sélection de la ligne avec l'index 1
print(df.loc[1])

print('################### Exercice 4 ###################')

import pandas as pd
import numpy as np

# Création du DataFrame
df = pd.DataFrame({
    'A': [1, 2, 3],
    'B': [4, 5, 6],
    'C': [7, 8, 9]
})

# Ajout de la colonne Sum
df['Sum'] = df['A'] + df['B'] + df['C']

# Suppression de la colonne Sum
df = df.drop(columns=['Sum'])

# Ajout de la colonne Random avec des valeurs aléatoires
df['Random'] = np.random.randn(len(df))

print(df)
print('################### Exercice 5 ###################')

import pandas as pd

# Création des DataFrames
left = pd.DataFrame({
    'key': [1, 2, 3],
    'A': ['A1', 'A2', 'A3'],
    'B': ['B1', 'B2', 'B3']
})

right = pd.DataFrame({
    'key': [1, 2, 3],
    'C': ['C1', 'C2', 'C3'],
    'D': ['D1', 'D2', 'D3']
})

# Fusion des DataFrames avec une jointure externe
merged_df = pd.merge(left, right, how='outer', on='key')

# Ajout de la colonne E au DataFrame de droite
right['E'] = ['E1', 'E2', 'E3']

# Mise à jour de la fusion pour inclure la nouvelle colonne
merged_df = pd.merge(left, right, how='outer', on='key')

print(merged_df)
print('################### Exercice 6 ###################')

import pandas as pd
import numpy as np

# Création du DataFrame
df = pd.DataFrame({
    'A': [1.0, np.nan, 7.0],
    'B': [np.nan, 5.0, 8.0],
    'C': [3.0, 6.0, np.nan]
})

# Remplacement des valeurs NaN par 0
df_filled = df.fillna(0)
print(df_filled)

# Remplacement des valeurs NaN par la moyenne de la colonne
df_filled_mean = df.apply(lambda x: x.fillna(x.mean()), axis=0)
print(df_filled_mean)

# Suppression des lignes contenant des valeurs NaN
df_dropped = df.dropna()
print(df_dropped)
print('################### Exercice 7 ###################')

import pandas as pd

# Création du DataFrame
df = pd.DataFrame({
    'Category': ['A', 'B', 'A', 'B', 'A', 'B'],
    'Value': [1, 2, 3, 4, 5, 6]
})

# Groupement par colonne Category et calcul de la moyenne de la colonne Value
grouped_mean = df.groupby('Category').mean()
print(grouped_mean)

# Calcul de la somme au lieu de la moyenne
grouped_sum = df.groupby('Category').sum()
print(grouped_sum)

# Comptage du nombre d'entrées dans chaque groupe
grouped_count = df.groupby('Category').count()
print(grouped_count)
print('################### Exercice 8 ###################')

import pandas as pd

# Création du DataFrame
df = pd.DataFrame({
    'Category': ['A', 'A', 'A', 'B', 'B', 'B'],
    'Type': ['X', 'Y', 'X', 'Y', 'X', 'Y'],
    'Value': [1, 2, 3, 4, 5, 6]
})

# Création d'un tableau croisé dynamique montrant la moyenne de Value pour chaque Category et Type
pivot_table_mean = df.pivot_table(values='Value', index='Category', columns='Type', aggfunc='mean')
print(pivot_table_mean)

# Modification du tableau croisé dynamique pour montrer la somme de Value au lieu de la moyenne
pivot_table_sum = df.pivot_table(values='Value', index='Category', columns='Type', aggfunc='sum')
print(pivot_table_sum)

# Ajout de marges pour montrer la moyenne totale pour chaque Category et Type
pivot_table_margins = df.pivot_table(values='Value', index='Category', columns='Type', aggfunc='mean', margins=True)
print(pivot_table_margins)
print('################### Exercice 9 ###################')

import pandas as pd
import numpy as np

# Création d'un DataFrame de séries temporelles avec une plage de dates
date_range = pd.date_range(start='2023-01-01', periods=6, freq='D')
df = pd.DataFrame({
    'Date': date_range,
    'Value': np.random.randn(6)
})

# Définir la colonne Date comme index du DataFrame
df.set_index('Date', inplace=True)

# Resampling des données pour calculer la somme pour chaque période de 2 jours
resampled_df = df.resample('2D').sum()
print(resampled_df)
print('################### Exercice 10 ###################')

import pandas as pd
import numpy as np

# Création du DataFrame
df = pd.DataFrame({
    'A': [1.0, 2.0, np.nan],
    'B': [np.nan, 5.0, 8.0],
    'C': [3.0, np.nan, 9.0]
})

# Interpolation des valeurs manquantes
df_interpolated = df.interpolate()
print(df_interpolated)

# Suppression des lignes contenant des valeurs NaN
df_dropped = df.dropna()
print(df_dropped)

print('################### Exercice 11 ###################')

import pandas as pd

# Création du DataFrame
df = pd.DataFrame({
    'A': [1, 2, 3],
    'B': [4, 5, 6],
    'C': [7, 8, 9]
})

# Calcul de la somme cumulative
cumsum_df = df.cumsum()
print(cumsum_df)

# Calcul du produit cumulatif
cumprod_df = df.cumprod()
print(cumprod_df)

# Application d'une fonction pour soustraire 1 de tous les éléments du DataFrame
subtracted_df = df.applymap(lambda x: x - 1)
print(subtracted_df)