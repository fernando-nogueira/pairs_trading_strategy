import pandas as pd

def get_px_close_data() -> pd.DataFrame:
    
    # Abre os dados de preço de fechamento
    df = pd.read_parquet('./data/fechamento.parquet')
    df = df.iloc[:, 1:]

    return df
    
def get_px_open_data() -> pd.DataFrame:
    
    # Abre os dados de preço de abertura
    df = pd.read_parquet('./data/abertura.parquet')
    df = df.iloc[:, 1:]
    
    return df
    
def get_volume_data() -> pd.DataFrame:
    
    # Abre os dados de volume
    df = pd.read_parquet('./data/volume.parquet')
    df = df.iloc[:, 1:]
    
    return df

def get_cdi_data() -> pd.DataFrame:
    
    # Abre os dados de CDI
    df = pd.read_parquet('./data/cdi.parquet')
    df = df[['CDI Acumulado']]
    
    return df

def get_ibovespa_data() -> pd.DataFrame:

    # Abre os dados de Ibovespa
    df_ibov = pd.read_excel('./data/ibovespa.xlsx')
    df_ibov = df_ibov.iloc[3:]
    df_ibov.columns = ['date', 'Ibovespa']
    df_ibov = df_ibov.set_index('date')

    return df_ibov