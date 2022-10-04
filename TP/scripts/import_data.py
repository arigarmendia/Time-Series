import pandas as pd

def data():
    #Importo Dataset consolidado
    molinetes = pd.read_csv('dataset/molinetes_consolidado.csv', index_col='fecha')
    #Importo serie suavizada con mediana móvil
    rolling_median = pd.read_csv('dataset/rolling_median.csv', index_col='fecha')
    #Importo serie diferenciada
    difference = pd.read_csv('dataset/difference.csv', index_col='fecha')
    return molinetes, rolling_median, difference