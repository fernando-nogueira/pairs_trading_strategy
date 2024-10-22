from src import data as dd
import pandas as pd
import strategy as st
import pickle

dict_data = st.load_data(2003)

def simulate_strat(params: dict) -> pd.DataFrame:

    list_backtest = st.backtest_maker(
    **params
    )

    df_bt_res = st.see_backtest_results(
        list_backtest,
        dict_data,
        False
    )
    df_bt_res_net = st.see_backtest_results(
        list_backtest,
        dict_data,
        True
    )

    df_ret = pd.concat([df_bt_res['port_ret_cdi'], df_bt_res_net['port_ret_cdi']], axis=1)
    df_ret.columns = ['Gross Returns', 'Net Returns']

    return df_ret

def add_cdi_and_ibov(df_ret: pd.DataFrame) -> pd.DataFrame:

    df_cdi = dd.get_cdi_data()
    df_cdi = df_cdi.replace('-', None).copy().dropna()
    df_cdi = df_cdi.astype(float).pct_change()


    df_ibov = pd.read_excel('./data/ibovespa.xlsx')
    df_ibov = df_ibov.iloc[3:]
    df_ibov.columns = ['date', 'Ibovespa']
    df_ibov = df_ibov.set_index('date')
    df_ibov = df_ibov.replace('-', None).copy().dropna()
    df_ibov = df_ibov.astype(float).pct_change()

    df_ret['cdi'] = df_cdi['CDI Acumulado']
    df_ret['ibovespa'] = df_ibov['Ibovespa']
    df_ret = df_ret.dropna()

    return df_ret

# Esse conjunto de listas é o equivalente as opções 1 e 2 da tabela de parâmetros simulados
all_data = []
list_train = [252*2, 252*3]
list_test = [126, 252]
list_stop = [7.5/100, 1000/100]
list_window_mean = [63, 126]
list_beta = [126, 252]
list_dp_entry = [2, 2.5]
list_dp_exit = [0.5, 0.75]

# O total é de 128 simulações, vou salvando os modelos na pasta "models" como em formato pickle como ex: "models/1.pickle"

tot = 128
idx = 0

for train in list_train:
    for test in list_test:
        for stop in list_stop:
            for window in list_window_mean:
                for beta in list_beta:
                    for dp_entry in list_dp_entry:
                        for dp_exit in list_dp_exit:
                            
                            idx = idx + 1
                            # preencho os parâmetros com as opções equivalentes da simulação enquanto os parâmetros não escolhidos 
                            params = {
                                'dict_data' : dict_data,
                                'train_window' : train,
                                'test_window' : test,
                                'n_volume' : 80, # 80 ativos mais negociados no período de formação
                                'max_number_pairs' : 20, # máximo 20 ativos no portfólio
                                'assets_in_only_one_side' : True,
                                'bt_params' : {'stop_loss' : stop ,
                                        'leverage' : 1,
                                        'window' : window,
                                        'window_beta' : beta,
                                        'dp_entry' : dp_entry,
                                        'dp_exit' : dp_exit,
                                        'dp_stop' : 4, # 4 desvios de choque
                                        'costs' : 0.17/100} # 17 bps
                            }
                                    
                            list_backtest = st.backtest_maker(
                                **params
                            )
                            
                            strat_data = {
                                'train' : train,
                                'test' : test,
                                'stop' : stop,
                                'window' : window,
                                'beta' : beta,
                                'dp_entry' : dp_entry,
                                'dp_exit' : dp_exit,
                                'params' : params,
                                'bt'  : list_backtest
                            }
                            
                            # pasta chamada "models" roda o modelo e salva em arquivo .pickle
                            with open(f'./results/models/{idx}.pickle', 'wb') as handle:
                                pickle.dump(strat_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
