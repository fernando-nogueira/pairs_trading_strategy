import pandas as pd

def rolling_walk_forward_analysis_windows(train_window: int, 
                                          test_window_step: int,
                                          df: pd.DataFrame) -> dict[str: list[tuple[int, int]]]:
    
    """
    Quebra dos dados de treinamento e teste com a janela inicial rolando ao longo do conjunto de dados

    # Exemplo
    
    # primeira iteração
    i = 1
    train_window = 252 
    test_window_step = 100

    # df.iloc[100 * 0:252 + 100 * 0] train (0:252)
    # df.iloc[252*1 + 1:252*1 + 1 + 100] test (253:353)

    """
    
    list_train_window = []
    list_test_window = []
    
    i = 1
    
    while True:
        
        train_window_init = test_window_step * (i-1)
        train_window_final = train_window + test_window_step * (i-1)
        
        test_window_init = train_window + test_window_step * (i-1) + 1
        test_window_final = test_window_init + test_window_step
        
        list_train_window.append((train_window_init, train_window_final))
        list_test_window.append((test_window_init-1, test_window_final-1))
        
        if test_window_final > len(df):
            break
            
        i = i + 1

    return {
        'train' : list_train_window,
        'test' : list_test_window 
    }
       
def anchor_walk_forward_analysis_windows(train_window: int, 
                                          test_window_step: int,
                                          df: pd.DataFrame) -> dict[str: list[tuple[int, int]]]:
    
    """
    Quebra dos dados de treinamento e teste com a janela inicial ancorada
    
    # Exemplo
    
    # primeira iteração
    i = 1
    train_window = 252 
    test_window_step = 100

    # df.iloc[0:252 + 100 * 0] train (0:252)
    # df.iloc[252*1 + 1:252*1 + 1 + 100] test (253:353)

    """
    
    list_train_window = []
    list_test_window = []
    
    i = 1
    
    while True:
        
        train_window_final = train_window + test_window_step * (i-1)
        
        test_window_init = train_window + test_window_step * (i-1) + 1
        test_window_final = test_window_init + test_window_step
        
        list_train_window.append((0, train_window_final))
        list_test_window.append((test_window_init, test_window_final))
        
        if test_window_final > len(df):
            break
            
        i = i + 1

    return {
        'train' : list_train_window,
        'test' : list_test_window 
    }