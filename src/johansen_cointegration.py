from statsmodels.tsa.vector_ar.vecm import coint_johansen
import pandas as pd

def johansen_coint_test(
        df_train: pd.DataFrame,
        asset_x: str,
        asset_y: str,
        significance_level=0.05) -> dict:
    
    """
    Função que realiza o procedimento de Johansen para testar a cointegração entre os ativos
    """

    series1, series2 = df_train[asset_y], df_train[asset_x]
    data = pd.concat([series1, series2], axis=1)
    coint_result = coint_johansen(data, det_order=0, k_ar_diff=1)
    asset1, asset2 = data.columns.tolist()
    # Obter estatísticas de teste e valores críticos
    trace_stat = coint_result.lr1

    if significance_level == 0.01:
        crit_val_index = 0
    elif significance_level == 0.05:
        crit_val_index = 1
    elif significance_level == 0.10:
        crit_val_index = 2
    else:
        raise ValueError("Nível de significância deve ser 0.01, 0.05, ou 0.10")
    
    critical_values = coint_result.cvt[:, crit_val_index]
    
    # Verificar se há cointegração ao nível de significância especificado
    is_cointegrated = trace_stat[0] > critical_values[0] and trace_stat[1] > critical_values[1]
    
    if is_cointegrated:
        # Obter beta normalizado e resíduos
        beta = coint_result.evec[0] / coint_result.evec[0, 0]
        resids = data.values @ beta
        if beta[-1] > 0 and beta[-1] < 3: # Checar se o beta é positivo e menor que 3
            return {
                    'pair': (asset1, asset2),
                    'reg': {'resid':  resids, 'beta' : [beta[-1]]},
                    'stat' : trace_stat[0]
                }
        else: 
            return None
    else:
        return None
