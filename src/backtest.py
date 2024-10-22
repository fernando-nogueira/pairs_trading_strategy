import pandas as pd
import numpy as np

def get_zscore(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    asset_x: str,
    asset_y: str,
    window: int,
    window_beta: int,
    dp_entry=float,
    dp_exit=float) -> pd.DataFrame:
    
    """
    Cria o spread e o spread normalizado, além de fazer as estimações em janela rolante do beta, média e desvio-padrão, os parâmetros
    de entrada e saída da operação
    """

    # 1. Juntar os dados de treino e teste (pra ter mais amostra no início dos dados de teste)
    df_train_test = pd.concat([df_train, df_test])[[asset_x, asset_y]]
    
    # 2. Calcular o beta rolling
    # 2.1. Covariância Rolling
    df_cov_rol = df_train_test.rolling(window_beta - window).cov()[asset_x].reset_index()
    df_cov_rol = df_cov_rol[df_cov_rol['level_1'] == asset_y].set_index('Data')[[asset_x]]
    df_cov_rol.columns = ['cov']
    
    # 3.2 Variância de Y
    df_var_rol = df_train_test.rolling(window_beta - window).var()[[asset_y]]
    df_var_rol.columns = ['var']

    # 3.3. Beta
    df_beta = pd.DataFrame({'beta' : df_cov_rol['cov'] / df_var_rol['var']})
    df_spread = (df_train_test[asset_y] - df_beta['beta'] * df_train_test[asset_x]).to_frame().shift()
    df_spread.columns = ['spread']
    
    # 4. Defasa os dados
    df_spread['beta'] = df_beta['beta'].shift()
    df_spread['mean_t-1'] = df_spread['spread'].rolling(window).mean()
    df_spread['std_t-1'] = df_spread['spread'].rolling(window).std()
    df_spread = df_spread.dropna()
    df_spread['zscore'] = (df_spread['spread'] - df_spread['mean_t-1']) / df_spread['std_t-1']

    # 5. Cria os parâmetros de entrada e saída da estratégia
    df_spread['mean'] = 0
    df_spread['upper'] = dp_entry
    df_spread['lower'] = dp_entry * -1

    df_spread['upper_stop'] = dp_exit
    df_spread['lower_stop'] = dp_exit * -1
    
    df_spread['asset_x'] = df_train_test[asset_x]
    df_spread['asset_y'] = df_train_test[asset_y]
    
    # 6. Pega apenas o índice dos dados de teste
    df_zscore = df_spread[df_spread.index.isin(df_test.index)]    
    
    return df_zscore

def create_backtest_position(
        df_zscore: pd.DataFrame,
        dp_stop: float) -> pd.DataFrame:
    
    """
    Cria as posições que o par pode assumir, comprado, vendido e neutro
    Também inclui o mecanismo de stopar caso passe o desvio-padrao de choques seja ativado
    """
    
    df_bt = df_zscore.copy()
    
    sell_cond = df_bt['zscore'] >= df_bt['upper']
    buy_cond = df_bt['zscore'] <= df_bt['lower']

    current_position = 0
    df_bt['position'] = np.nan
    
    # Normalização para iniciarem na mesma escala (1)
    df_bt['asset_x'] = df_bt['asset_x'] / df_bt['asset_x'].iloc[0]
    df_bt['asset_y'] = df_bt['asset_y'] / df_bt['asset_y'].iloc[0]
    
    for i, row in df_bt.iterrows():

        if row['zscore'] >= dp_stop or row['zscore'] <= -dp_stop:
            # Condição de parada nos dados caso o stop de desvio-padrão (choques) seja ativado
            break
                
        if sell_cond[i]:
            current_position = -1

        elif buy_cond[i]:
            current_position = 1

        elif current_position == -1 and df_bt['zscore'][i] >= df_bt['upper_stop'][i]:
            # condição de saída da operação short
            current_position = -1

        elif current_position == 1 and df_bt['zscore'][i] <= df_bt['lower_stop'][i]:
            # condição de saída da operação
            current_position = 1
        
        else:
            current_position = 0

        df_bt.at[i, 'position'] = current_position

    return df_bt

def create_backtest_returns(
        df_bt: pd.DataFrame,
        leverage: float=1,
        stop_loss: float=0.1,
        costs: float=0.0017
    ) -> pd.DataFrame:

    """
    Cria o retorno da estratégia de acordo com o vetor de posições, calculando o lucro da ponta longa e da ponta short 
    """
    
    # Long and Short, comprado 1 real no ativo Y e 1 real no ativo X (valor financeiro igual, como os preços são normalizados ambos começam na mesma escala)

    df_pos = df_bt.copy()
    df_pos['position'] = df_pos['position'].replace(np.nan, 0)        
    
    weight = leverage / 2
    df_pos['long'] = df_pos['asset_y'] * weight # QUANTIDADE FIXA 
    df_pos['short'] = df_pos['asset_x'] * -weight # QUANTIDADE FIXA
    
    df_pos['long_gains'] = df_pos['long'].diff()
    df_pos['short_gains'] = df_pos['short'].diff()
    df_pos['gains'] = df_pos['long_gains'] + df_pos['short_gains']
    df_pos['gross_gains'] = df_pos['gains'].fillna(value=0).cumsum() + leverage
    
    df_pos['portfolio'] = (df_pos['gross_gains'] / leverage).pct_change().fillna(value=0)
    df_pos['strategy'] = df_pos['portfolio'] * df_pos['position']
    
    # os custos só entram quando o vetor de posição muda, se eu me mantenho comprado eu nao pago custo, 
    # eu só pago se eu sair de comprado para vendido por exemplo por isso eu faço o diff no vetor de posições
    
    df_pos['strategy_net'] = df_pos['strategy'] - (abs(df_pos['position'].diff()).shift() * costs)     
    df_pos['strategy_acum'] = (df_pos['strategy'] + 1).cumprod() - 1
    df_pos['strategy_acum_net'] = (df_pos['strategy_net'] + 1).cumprod() - 1
    
    def adjust_for_stop_loss(df_pos: pd.DataFrame, col_acum: str, col_ret: str, stop_loss: float) -> pd.DataFrame:
        
        """
        Função para ajustar o resultado da estratégia com o stop loss (rentabilidade acumulada) passado
        """
        
        df_stop_loss = df_pos[df_pos[col_acum] <= -stop_loss]
        
        if len(df_stop_loss) > 0:
            time_stop_loss = df_stop_loss.iloc[0].name
            
            # Dataframe Antes do Stop
            df2 = df_pos[df_pos.index > time_stop_loss].copy()
            
            # Dataframe Depois do Stop
            df1 = df_pos[df_pos.index <= time_stop_loss].copy()
            
            df2[col_ret] = 0
            df2['position'] = 0
            
            df_pos = pd.concat([df1, df2])
            df_pos[col_acum] = (1 + df_pos[col_ret]).cumprod() - 1

        return df_pos
    
    df_pos = adjust_for_stop_loss(df_pos, 'strategy_acum', 'strategy', stop_loss)
    df_pos = adjust_for_stop_loss(df_pos, 'strategy_acum_net', 'strategy_net', stop_loss)

    return df_pos