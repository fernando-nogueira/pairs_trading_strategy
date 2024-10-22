import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import strategy as st
import datetime as dt
import pandas as pd
import numpy as np
import pickle
import os

"""

Gráficos / Tabelas apresentados no capítulo de resultados empíricos

"""

dict_data = st.load_data(2003)

def see_returns(list_backtest) -> pd.DataFrame:
    
    """
    Função para visualizar os retornos da estratégia
    """
    
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

def add_ibov(df_ret: pd.DataFrame) -> pd.DataFrame:

    """
    Função para visualizar os retornos da estratégia contra o Ibovespa
    """
    
    df_ibov = pd.read_excel('./data/ibovespa.xlsx')
    df_ibov = df_ibov.iloc[3:]
    df_ibov.columns = ['date', 'Ibovespa']
    df_ibov = df_ibov.set_index('date')
    df_ibov = df_ibov.replace('-', None).copy().dropna()
    df_ibov = df_ibov.astype(float).pct_change()

    df_ret['ibovespa'] = df_ibov['Ibovespa']
    df_ret = df_ret.dropna()

    return df_ret

# Lê todos os arquivos .pickle que estão na pasta de modelos
path = './results/models/'
list_path = [path + x for x in os.listdir(path)]

df_models = pd.DataFrame()
# Nome da estratégia modelo
model = f'train=504,test=252,stop=0.075,window=126,beta=252,dp_entry=2.5,dp_exit=0.75'

list_rows = []
# Lê todos os arquivos Pickle
for p in list_path:
    
    with open(p, 'rb') as handle:
        dict_pickle = pickle.load(handle)
    
    train = dict_pickle['train']
    test = dict_pickle['test']
    stop = dict_pickle['stop']
    window = dict_pickle['window']
    beta = dict_pickle['beta']
    dp_entry = dict_pickle['dp_entry']
    dp_exit = dict_pickle['dp_exit']

    name_model = f'train={train},test={test},'\
                 f'stop={stop},window={window}'\
                 f',beta={beta},dp_entry={dp_entry}'\
                 f',dp_exit={dp_exit}'
    
    
    list_backtest = dict_pickle['bt']
    df_ret = see_returns(list_backtest)
    df_models[name_model] = df_ret['Net Returns']

    if name_model == model:
        bt_ = dict_pickle['bt']
        for i in bt_:
            for y in i:
                pair = y['pair']
                st = y['start_date_test']
                ed = y['end_date_test']
                ret = y['df_pos']['strategy_acum_net'].iloc[-1]
                list_rows.append({'model' : p, 'pair' : pair, 'st' : st, 'ed' : ed, 'ret': ret})

# Junta todos as operações dos modelos nesse dataframe
df_operations = pd.DataFrame(list_rows)
df_operations['pair_x'] = [x['pair_x'] for x in df_operations['pair']]
df_operations['pair_y'] = [x['pair_y'] for x in df_operations['pair']]
df_operations['pair'] = df_operations['pair'].astype(str)

# Agrupamento por pares para criar as estatísticas
df_qtt = df_operations.groupby(['pair_x', 'pair_y'])['ret'].count().reset_index().sort_values('ret', ascending=False).head(20)

# Quantidade de vezes que aparece, retorno máximo e mínimo das operações
list_ops = []
for _, row in df_qtt.iterrows():
    df_op = df_operations[(df_operations['pair_x'] == row['pair_x']) & (df_operations['pair_y'] == row['pair_y'])]
    op = {'x' : row['pair_x'], 'y' : row['pair_y'],
          'max': df_op['ret'].max(), 'min' : df_op['ret'].min(),
          'mean' : df_op['ret'].mean(), 'qtd' : row['ret']}
    
    list_ops.append(op)

# Tabela 4
pd.DataFrame(list_ops)

# Coloca todos os modelos para começar em janeiro de 2006 e terminar em dezembro de 2023
df_models = df_models[(df_models.index >= dt.datetime(2006, 1, 1)) & (df_models.index < dt.datetime(2024, 1, 1))]
df_models_cp = (1 + df_models.dropna()).cumprod() - 1

# Adiciona o Ibovespa para efeitos de comparação
df_models_ = add_ibov(df_models)
df_models_cp_ = (1 + df_models_).cumprod() - 1

# Modelo aleatório das simulações para que a legenda do nome "Simulações" aparece apenas uma vez
random_model = 'train=756,test=126,stop=0.075,window=126,beta=252,dp_entry=2,dp_exit=0.75'

df_models_cp_ = df_models_cp_[pd.DatetimeIndex(df_models_cp.index).year <= 2023]

fig, ax = plt.subplots(figsize=(10, 3.5), dpi=300)
others = [model, 'ibovespa']
for x in [x for x in df_models_cp.columns.tolist() if x not in others] + others:

    if x == model:
        ax.plot(df_models_cp[x], color='#003566', linewidth=2, label='Modelo')
    elif x == 'ibovespa':
        ax.plot(df_models_cp[x], color='#0466c8', linewidth=2, label='Ibovespa')
    elif x == random_model:
        ax.plot(df_models_cp[x], color='#979dac', linewidth=0.75, label='Simulações')
    else:
        ax.plot(df_models_cp[x], color='#979dac', linewidth=0.75)

ax.xaxis.set_major_formatter(
    mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
ax.grid(axis='x', linewidth=0.5)
plt.legend()

# Figura 6. Rentabilidade acumulada das estratégias simuladas comparadas ao Ibovespa
plt.show()

# dataframe dos retornos acumulados na última data disponível
df_last_values = df_models_cp.iloc[-1].to_frame().T

# QTD de estratégias ganhadoras (grupo dos ganhadores)
qtt_win = len(df_last_values.T[df_last_values.T > df_last_values['ibovespa']].dropna())

# QTD de estratégias perdedoras (grupo dos perdedores)
qtt_loss = len(df_last_values.T[df_last_values.T < df_last_values['ibovespa']].dropna())

# de onde vem os 31,25%
qtt_win / 128

# Cálculo da volatilidade anualziada em janela rolante de 252 dias 
df_models_vol = df_models_.copy()
df_models_vol = (df_models_vol.rolling(252).std() * np.sqrt(252)).dropna()

fig, ax = plt.subplots(figsize=(10, 3.5), dpi=300)

for x in df_models_vol.columns.tolist():

    if x == model:
        ax.plot(df_models_vol[x], color='#003566', linewidth=2, label='Modelo')
    elif x == 'ibovespa':
        ax.plot(df_models_vol[x], color='#0466c8', linewidth=2, label='Ibovespa')
    elif x == random_model:
        ax.plot(df_models_vol[x], color='#979dac', linewidth=0.75, label='Simulações')
    else:
        ax.plot(df_models_vol[x], color='#979dac', linewidth=0.75)

ax.xaxis.set_major_formatter(
    mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
ax.grid(axis='x', linewidth=0.5)
plt.legend()
plt.show()

# Cálculo da correlação em janela de 252 dias dos retornos com o ibovespa

df_corr = df_models_.rolling(252).corr()[['ibovespa']].reset_index().pivot_table(index='Data', values='ibovespa', columns='level_1').dropna()
df_corr = df_corr[[x for x in df_corr.columns if x != 'ibovespa']]

fig, ax = plt.subplots(figsize=(10, 3.5), dpi=300)

for x in df_corr.columns.tolist():

    if x == model:
        ax.plot(df_corr[x], color='#003566', linewidth=2, label='Modelo')
    elif x == random_model:
        ax.plot(df_corr[x], color='#979dac', linewidth=0.75, label='Simulações')
    else:
        ax.plot(df_corr[x], color='#979dac', linewidth=0.75)

ax.xaxis.set_major_formatter(
    mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
ax.grid(axis='x', linewidth=0.5)
plt.legend()
# Figura 9 - Correlação em uma janela de 252 dias das estratégias simuladas com o Ibovespa
plt.show()


# Cálculo do Drawdown
df_mdd = (df_models_ + 1).cumprod() / (df_models_ + 1).cumprod().cummax() - 1

fig, ax = plt.subplots(figsize=(10, 3.5), dpi=300)

others = [model, 'ibovespa']
for x in [x for x in df_mdd.columns.tolist() if x not in others] + others:

    if x == model:
        ax.plot(df_mdd[x], color='#003566', linewidth=2, label='Modelo')
    elif x == 'ibovespa':
        ax.plot(df_mdd[x], color='#0466c8', linewidth=2, label='Ibovespa')
    elif x == random_model:
        ax.plot(df_mdd[x], color='#979dac', linewidth=0.75, label='Simulações')
    else:
        ax.plot(df_mdd[x], color='#979dac', linewidth=0.75)

ax.xaxis.set_major_formatter(
    mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
ax.grid(axis='x', linewidth=0.5)
plt.legend()
# Figura 8 - Drawdown das estratégias simuladas e do Ibovespa
plt.show()

# Cálculo do retorno acumulado em janelas móveis de 12 meses
df_ret_rolling = df_models_[[x for x in df_models_.columns if x != 'cdi']].copy()
others = [model, 'ibovespa']

df_ret_rolling = (1 + df_ret_rolling).rolling(window=252).apply(np.prod, raw=True) - 1
df_ret_rolling = df_ret_rolling.dropna()

df_time_above = df_ret_rolling[[x for x in df_ret_rolling.columns if x != 'ibovespa']].apply(lambda x: x > df_ret_rolling['ibovespa'])

pct_rolling = df_time_above[df_time_above == True].fillna(value=0).sum() / len(df_time_above)

fig, ax = plt.subplots(figsize=(10, 3.5), dpi=300)
for x in [x for x in df_ret_rolling.columns.tolist() if x not in others] + others:
    
    if x == model:
        ax.plot(df_ret_rolling[x], color='#003566', linewidth=2, label='Modelo')
    elif x == 'ibovespa':
        ax.plot(df_ret_rolling[x], color='#0466c8', linewidth=2, label='Ibovespa')
    elif x == random_model:
        ax.plot(df_ret_rolling[x], color='#979dac', linewidth=0.75, label='Simulações')
    else:
        ax.plot(df_ret_rolling[x], color='#979dac', linewidth=0.75)

ax.xaxis.set_major_formatter(
    mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
ax.grid(axis='x', linewidth=0.5)
plt.legend()
# Figura 10. Retorno acumulado em janelas móveis de 12 meses 
plt.show()
plt.savefig('./charts/analysis_fig5.jpeg',  dpi=300)

# modelos ganhadores
mwin = df_last_values.T[df_last_values.T > df_last_values['ibovespa']].dropna().index.tolist()
# models perdedores
mloss = df_last_values.T[df_last_values.T < df_last_values['ibovespa']].dropna().index.tolist()


# resultado
mwin_rent = (1 + df_models_cp[mwin].iloc[-1].mean()) ** (252/ len(df_models_cp)) -1
mloss_rent = (1 + df_models_cp[mloss].iloc[-1].mean()) ** (252/ len(df_models_cp)) -1
ibov_rent = (1 + df_models_cp['ibovespa'].iloc[-1].mean()) ** (252/ len(df_models_cp)) -1
model_rent = (1 + df_models_cp[model].iloc[-1].mean()) ** (252/ len(df_models_cp)) -1

# volatilidade
mwin_vol = df_models_vol[mwin].iloc[-1].mean()
mloss_vol = df_models_vol[mloss].iloc[-1].mean()
ibov_vol = df_models_vol['ibovespa'].iloc[-1].mean()
model_vol = df_models_vol[model].iloc[-1].mean()

# corr
mwin_corr = df_corr[mwin].iloc[-1].mean()
mloss_corr = df_corr[mloss].iloc[-1].mean()
model_corr = df_corr[model].iloc[-1].mean()

# drawdown
mwin_mdd = df_mdd[mwin].min().mean()
mloss_mdd = df_mdd[mloss].min().mean()
ibov_mdd = df_mdd['ibovespa'].min().mean()
model_mdd = df_mdd[model].min().mean()


# % de vezes acima do benchmark
win_rol = pct_rolling[pct_rolling.index.isin(mwin)].mean()
loss_rol = pct_rolling[pct_rolling.index.isin(mloss)].mean()
model_rol = pct_rolling[pct_rolling.index.isin([model])].mean()


table = pd.DataFrame({
    'Rentabilidade Anualizada' : [mwin_rent, mloss_rent, ibov_rent, model_rent],
    'Volatilidade Anualizada' : [mwin_vol, mloss_vol, ibov_vol, model_vol],
    'Correlação' : [mwin_corr, mloss_corr, 1, model_corr],
    'Máximo Drawdown' : [mwin_mdd, mloss_mdd, ibov_mdd, model_mdd],
    '% Acima do Ibovespa' : [win_rol, loss_rol, 1, model_rol]
}, index=['Ganhadores', 'Perdedores', 'Ibovespa', 'Modelo'])

# Tabela 2
print(table)

list_dict_pct = []
last_result_ibov = df_models_cp['ibovespa'].iloc[-1]

dict_param = {
    'train=' : [504, 756],
    'test=' : [126, 252],
    'stop=' : [7.50/100, 1000/100],
    'window=' : [63, 126],
    'beta=' : [126, 252],
    'dp_entry=' : [2, 2.5],
    'dp_exit=' : [0.5, 0.75],
}

for param in dict_param:
    list_values = dict_param[param]
    dict_pct = {}
    for i, value in enumerate(list_values):

        df_model_value = df_models_cp[[x for x in df_models_cp.columns if f'{param}{value}' in x]]
        last_result_value = df_model_value.iloc[-1]
        last_result_value_bench = last_result_value[last_result_value > last_result_ibov]
        pct = len(last_result_value_bench) / len(last_result_value)
        dict_pct[i + 1] = pct

    list_dict_pct.append(dict_pct)

# Tabela 3
df_pct = pd.DataFrame(list_dict_pct)
table_adj = df_pct.div(df_pct.sum(axis=1), axis=0) * 0.3125 # % de estratégias acima do Ibovespa
table_adj.index = dict_param.keys() 