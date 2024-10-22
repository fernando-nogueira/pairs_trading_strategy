import johansen_cointegration as joh
from src import backtest as bt 
from typing import Callable
from src import pairs
from src import data
from src import wfa
import pandas as pd
import numpy as np

def load_data(year: int) -> list[pd.DataFrame]:
    
    """
    Carrega todos os conjuntos de dados
    """

    df_clos = data.get_px_close_data()
    df_volm = data.get_volume_data()
    df_cdi = data.get_cdi_data()
    df_ibov = data.get_ibovespa_data()

    df_cl = df_clos[df_clos.index.year >= year].copy()
    df_vl = df_volm[df_volm.index.year >= year].copy()
    df_cdi = df_cdi[df_cdi.index.year >= year].copy()
    df_ibov = df_ibov[df_ibov.index.year >= year].copy()

    df_cl = df_cl.replace('-', None).dropna(axis=0, how='all').astype(float)
    df_vl = df_vl.replace('-', None).dropna(axis=0, how='all').astype(float)
    df_cdi = df_cdi.replace('-', None).dropna(axis=0, how='all').astype(float)
    df_ibov = df_ibov.replace('-', None).dropna(axis=0, how='all').astype(float)

    dict_data = {
        'dados_de_fechamento' : df_cl,
        'dados_de_volume' : df_vl,
        'dados_do_cdi' : df_cdi,
        'dados_do_ibovespa' : df_ibov
    }
    
    return dict_data

def get_top_mean_volume_assets(
    df_vl: pd.DataFrame,
    period_index: pd.DatetimeIndex,
    traded_assets: list[str],
    n: int) -> list[str]:
    
    """
    Pega os N ativos com maior média de volume no período
    """
    
    df_vol_train = df_vl[df_vl.index.isin(period_index)][traded_assets].astype(float).copy()
    rank_volume = df_vol_train.mean().sort_values(ascending=False).iloc[0:n].index
    
    return rank_volume

def get_pairs_cointegration_test(
    df_pairs_combination: pd.DataFrame,
    df_close_train_norm: pd.DataFrame,
    max_number_pairs: int, 
    cointegration_test_function: Callable,
    asset_in_only_one_side: bool):
    
    """
    Dado a combinação de todos os pares retorna apenas os "n" primeiros pares cointegrados
    e ranqueados pelo valor da estatística de traço de acordo com o parâmetro "max_number_pairs"
    o parâmetro "asset_in_only_one_side" assume uma variável booleana, se for True
    o par pode assumir apenas uma ponta dos pares resolvendo a questão de Law et al (2017)
    A função de cointegração também é um parâmetro para o código ser genérico, mas na prática
    é o procedimento de Johansen criado em johansen_cointegration.py
    """
    
    list_pair_info = []

    for pair_x, pair_y in zip(df_pairs_combination['asset_1'], df_pairs_combination['asset_2']):
        
        dict_pair = cointegration_test_function(df_close_train_norm, pair_x, pair_y)
        
        if dict_pair != None:
            list_pair_info.append(dict_pair)

        
    if len(list_pair_info) > 0:
        df_pairs = pd.json_normalize([x for x in list_pair_info if not isinstance(x, type(None))])    
        df_pairs['pair_y'] = df_pairs['pair'].str[0]
        df_pairs['pair_x'] = df_pairs['pair'].str[1]
        df_pairs['beta'] = df_pairs['reg.beta'].str[0]

        if asset_in_only_one_side:
            stocks_added = set()
            
            def asset_in_only_one_pair(row):
                if row['pair_y'] in stocks_added or row['pair_x'] in stocks_added:
                    return False
                
                stocks_added.add(row['pair_y'])
                stocks_added.add(row['pair_x'])
                
                return True

            df_pairs = df_pairs[df_pairs.apply(asset_in_only_one_pair, axis=1)]
            
        df_pairs = df_pairs[['pair_y', 'pair_x', 'stat', 'beta', 'pair']].sort_values('stat', ascending=False).reset_index(drop=True)
        
        if len(df_pairs) > max_number_pairs:
            df_pairs = df_pairs.iloc[:max_number_pairs]
        
        return df_pairs

def set_pairs_assets_in_close_data(
    df_close_test: pd.DataFrame,
    df_pairs: pd.DataFrame) -> pd.DataFrame:
    
    """
    Volta o dataframe de teste com os pares selecionados pela função anterior
    """
    uniq_asset = pd.Series(df_pairs['pair_y'].tolist() + df_pairs['pair_x'].tolist()).unique()
    df_close_test = df_close_test[uniq_asset]

    return df_close_test

def get_backtest_results(df_pairs: pd.DataFrame,
                            df_close_train: pd.DataFrame,
                            df_close_test: pd.DataFrame,
                            bt_params: dict[str: float]) -> list[dict[str]]:
    
    """
    Com os parâmetros da estratégia passados pelo dicionário "bt_params"
    Pega os resultados do backtest em uma lista, escolhi a estrutura de lista
    com as informações em dicionário para ter uma granulidade maior das operações
    e informações de cada backtest 
    """
    
    list_bt_results = []
    for _, row in df_pairs.iterrows():
        
        df_zscore = bt.get_zscore(
            df_train=df_close_train,
            df_test=df_close_test,
            asset_x=row['pair_x'],
            asset_y=row['pair_y'],
            window=bt_params['window'],
            window_beta=bt_params['window_beta'],
            dp_entry=bt_params['dp_entry'],
            dp_exit=bt_params['dp_exit']
        )
        
        df_bt = bt.create_backtest_position(
            df_zscore=df_zscore,
            dp_stop=bt_params['dp_stop']
        )
        df_pos = bt.create_backtest_returns(
            df_bt,
            leverage=bt_params['leverage'],
            stop_loss=bt_params['stop_loss'],
            costs=bt_params['costs']
        )
        
        bt_results = {
            'pair' : {'pair_x' : row['pair_x'], 'pair_y' : row['pair_y']},
            'df_zscore' : df_zscore,
            'df_bt' : df_bt,
            'df_pos' : df_pos,
            'params' : bt_params,
            'start_date_train' : df_close_train.index[0],
            'end_date_train' : df_close_train.index[-1],
            'start_date_test' : df_close_test.index[0],
            'end_date_test' : df_close_test.index[-1]
        }
        
        list_bt_results.append(bt_results)
    
    return list_bt_results
    
        
def get_backtest_returns(list_bt_results: list[dict[str]], net: bool) -> pd.DataFrame:
    
    """
    Pega o retorno da estratégia pela lista de backtest criada pela função anterior
    net é uma variável booleana, se for verdade pega a coluna do retorno líquido
    isto é, tirando os custos de operações, se não pega a rentabilidade bruta
    """
    
    if net:
        strat_col = 'strategy_net'
    else:
        strat_col = 'strategy'
    list_df_bt = []
    for bt_res in list_bt_results:
        df_bt = bt_res['df_pos'][strat_col].to_frame().copy()
        p = bt_res['pair']
        df_bt.columns = [f"{p['pair_x']} x {p['pair_y']}"]
        list_df_bt.append(df_bt)
    
    df_bt_ret = pd.concat(list_df_bt, axis=1)            
    
    return df_bt_ret

def add_cdi_into_no_trading_days(df_bt_ret: pd.DataFrame, df_cdi: pd.DataFrame) -> pd.DataFrame:

    """
    Adiciona o retorno do CDI em dias "Não operados" pela estratégia, isto é, ou se o par for stopado
    ou se o vetor de posições estiver indicando "Neutro" e consequentemente for 0
    """

    df_bt_ret_cdi = df_bt_ret.replace(0, np.nan).copy()
    df_cdi_replace = df_cdi.pct_change()[df_cdi.columns[0]]
    
    for column in df_bt_ret_cdi:
        df_bt_ret_cdi[column] = df_bt_ret_cdi[column].fillna(df_cdi_replace)

    return df_bt_ret_cdi

def backtest_maker(
    dict_data: dict[str: pd.DataFrame],
    train_window: int,
    test_window: int,
    n_volume: int,
    max_number_pairs: int,
    assets_in_only_one_side: bool,
    bt_params: dict[str: float]):
    
    """
    Função que faz o backtest da estratégia de todos os períodos
    A função anterior de backtest faz apenas para um período, aqui faz de todos os períodos
    dado os parâmetros explicitados e o conjunto de dados utilizado
    """
    
    df_cl = dict_data['dados_de_fechamento']
    df_vl = dict_data['dados_de_volume']
    df_cdi = dict_data['dados_do_cdi']
    
    dict_wfa = wfa.rolling_walk_forward_analysis_windows(
        train_window,
        test_window,
        df_cl
    )
    
    list_backtest = []

    for train, test in zip(dict_wfa['train'], dict_wfa['test']):
               
        df_close_train = df_cl.iloc[train[0]:train[1]].copy()
        df_close_test = df_cl.iloc[test[0]:test[1]].copy()
                
        # 1. Os ativos devem ter sido negociados todos os dias do período
        df_close_train = df_close_train.dropna(axis=1, how='any')
        
        # 2. Pega os N ativos com maior volume
        rank_volume = get_top_mean_volume_assets(
            df_vl,
            period_index=df_close_train.index,
            traded_assets=df_close_train.columns,
            n=n_volume
        )
        
        # 3. Os ativos no ranque de volume são normalizados no dataset
        df_close_train = df_close_train[rank_volume]
        df_close_train_norm = df_close_train.copy()
        
        # 4. Faz a combinação de todos os ativos possíveis
        df_pairs_combination = pairs.get_all_pairs_combination(df_close_train)
        
        # 5. Realiza os testes de cointegração nos pares de ativos
        df_pairs = get_pairs_cointegration_test(
            df_pairs_combination,
            df_close_train_norm,
            max_number_pairs=max_number_pairs, 
            cointegration_test_function=joh.johansen_coint_test,
            asset_in_only_one_side=assets_in_only_one_side)
        
        print('Pares cointegrados obtidos')
        if isinstance(df_pairs, pd.DataFrame):

            # 6. Pega os ativos dos pares cointegrados e filtra o dataset de fechamento de preço para conter apenas esses ativos
            df_close_train = set_pairs_assets_in_close_data(df_close_train, df_pairs)
            df_close_test = set_pairs_assets_in_close_data(df_close_test, df_pairs).ffill()
            
            # 7. Backtest da estratégia com os parâmetros
            list_bt_results = get_backtest_results(
                df_pairs,
                df_close_train,
                df_close_test,
                bt_params=bt_params
            )
            
            df_bt_ret = get_backtest_returns(list_bt_results, False)
            df_bt_ret_net = get_backtest_returns(list_bt_results, True)
            df_bt_ret_cdi = add_cdi_into_no_trading_days(df_bt_ret, df_cdi)
            df_bt_ret_cdi_net = add_cdi_into_no_trading_days(df_bt_ret_net, df_cdi)

            list_backtest.append(list_bt_results)
            
            print(f"WFA: {dict_wfa['train'].index(train) + 1} / {len(dict_wfa['train'])}\n",
                f'Gross: \n',
                f"Retorno: {round(((1 + df_bt_ret_cdi).cumprod()-1).mean(axis=1).iloc[-1],2)* 100}% c/ CDI\n",
                f"Retorno: {round(((1 + df_bt_ret).cumprod()-1).mean(axis=1).iloc[-1],2)* 100}% s/ CDI\n,",
                f'Net:\n',
                f"Retorno: {round(((1 + df_bt_ret_cdi_net).cumprod()-1).mean(axis=1).iloc[-1],2)* 100}% c/ CDI\n",
                f"Retorno: {round(((1 + df_bt_ret_net).cumprod()-1).mean(axis=1).iloc[-1],2)* 100}% s/ CDI")

        else:
            print(f"WFA: {dict_wfa['train'].index(train) + 1} / {len(dict_wfa['train'])}\n",
                f"Nenhum par encontrado")

    return list_backtest
    

def see_backtest_results(list_backtest: list,
                         dict_data: dict[str: pd.DataFrame],
                         net: bool) -> pd.DataFrame:
    
    """
    Função auxiliar para visualizar os resultados da estratégia de uma lista de diferentes períodos de backtest
    """
    df_cdi = dict_data['dados_do_cdi']
    
    list_df = []

    for list_bt_results in list_backtest:
        df_bt_ret = get_backtest_returns(list_bt_results, net)
        df_bt_ret_cdi = add_cdi_into_no_trading_days(df_bt_ret, df_cdi)

        # A média é dos retornos acumulados é para fazer o portfólio equal weight (pesos iguais) a nível de retorno acumulado
        # e o pct_change() são os retornos desse portfólio

        df_port_ret = (((1 + df_bt_ret).cumprod()).mean(axis=1)).pct_change()
        df_port_ret_cdi =  (((1 + df_bt_ret_cdi).cumprod()).mean(axis=1)).pct_change()
        
        df_bt_res = pd.DataFrame({
            'port_ret' : df_port_ret,
            'port_ret_cdi' : df_port_ret_cdi
        })
        
        list_df.append(df_bt_res)

    df_bt_results = pd.concat(list_df)

    return df_bt_results