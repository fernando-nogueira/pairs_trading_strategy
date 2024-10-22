import pandas as pd

def get_all_pairs_combination(df: pd.DataFrame) -> pd.DataFrame:
    
    # Pega todas as combinações de pares possiveis
    list_pairs = []
    for pair_x in df.columns:
        for pair_y in df.columns:
            # não queremos pares repetidos nem ativos iguais como pares
            if pair_x != pair_y and \
                (pair_x, pair_y) not in list_pairs and\
                (pair_y, pair_x) not in list_pairs:
                
                list_pairs.append((pair_x, pair_y))

    n = len(df.columns)
    assert len(list_pairs) == (n * (n-1))/2 # Huck e Afawubo (2014)

    df_pairs = pd.DataFrame(
        list_pairs,
        columns=['asset_1', 'asset_2']
    )
    
    return df_pairs