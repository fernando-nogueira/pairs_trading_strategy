import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import src.backtest as bt
import pandas as pd
import strategy as m
import locale

"""
Gráficos apresentados no capítulo de metodologia

"""

# Dados a partir do ano de 2016
dict_data = m.load_data(2016)

# Seleção dos pares ITUB4 e BBDC4
asset_x = 'ITUB4'
asset_y = 'BBDC4'

# Criação dos dados de treinamento e teste
df_train = dict_data['dados_de_fechamento'][[asset_x, asset_y]].iloc[:252]
df_test = dict_data['dados_de_fechamento'][[asset_x, asset_y]].iloc[252:252*2]

# Normalização dos preços de fechamento
df_one = df_train / df_train.iloc[0]

locale.setlocale(locale.LC_ALL, 'pt_pt.UTF-8')

# Gráfico 1 - Figura 1 Preço de fechamento normalizado entre ITUB4 e BBDC4
fig, ax = plt.subplots(figsize=(10,3.5), dpi=300)
ax.plot(df_one[asset_x], label=asset_x, color='#0466c8')
ax.plot(df_one[asset_y], label=asset_y, color='#979dac')
ax.xaxis.set_major_formatter(
    mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
ax.grid(axis='x', linewidth=0.5)
ax.legend()
plt.show()
# plt.savefig('./charts/fig1.jpeg',  dpi=300)

bt_params = {
    'stop_loss' : 5 / 100 ,
    'leverage' : 1.5,
    'window' : 63,
    'window_beta' : 252,
    'dp_entry' : 2.5,
    'dp_exit' : 0.75,
    'dp_stop' : 4
}

df_train = dict_data['dados_de_fechamento'][[asset_x, asset_y]].iloc[:252]
df_train2 = dict_data['dados_de_fechamento'][[asset_x, asset_y]].iloc[252:252*2]
df_test = dict_data['dados_de_fechamento'][[asset_x, asset_y]].iloc[252*2:252*3]
df_one = df_train / df_train.iloc[0]

df_zscore = bt.get_zscore(
    df_train=df_train,
    df_test=df_train2,
    asset_x=asset_x,
    asset_y=asset_y,
    window=bt_params['window'],
    window_beta=bt_params['window_beta'],
    dp_entry=bt_params['dp_entry'],
    dp_exit=bt_params['dp_exit']
)


fig, ax = plt.subplots(figsize=(10,3.5), dpi=300)
# Gráfico 2 - Figura 2 Spread normalizado com bandas de saída da operação

data = df_zscore[['zscore', 'upper', 'lower', 'upper_stop', 'lower_stop']]

ax.plot(data['zscore'], color='#0466c8', label='Z-Score')
ax.plot(data['upper'], color='#979dac', label='Limite Superior', linestyle='--')
ax.plot(data['lower'], color='#979dac', label='Limite Inferior', linestyle='--')
ax.plot(data['upper_stop'], color='#979dac', label='', linestyle='.')
ax.plot(data['lower_stop'], color='#979dac', label='', linestyle='.')
        
ax.fill_between(data.index, data['upper_stop'], data['lower_stop'], 
                where=data['upper'] > data['lower'],
                facecolor='grey', alpha=0.3, label='Equilíbrio')

plt.xlabel('Data')
ax.xaxis.set_major_formatter(
    mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
ax.grid(axis='x', linewidth=0.5)
ax.legend()
plt.show()

df_bt = bt.create_backtest_position(
    df_zscore=df_zscore,
    dp_stop=bt_params['bt_params']['dp_stop']
)

fig, ax = plt.subplots(figsize=(10,3.5), dpi=300)

# Gráfico 3 - Figura 3 Posições compradas (1), vendidas (-1) e neutras (0) do par

ax.plot(df_bt['position'], color='#0466c8', label='Posição')
plt.xlabel('Data')
ax.xaxis.set_major_formatter(
    mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
ax.grid(axis='x', linewidth=0.5)
ax.legend()
plt.show()

# Sem mecanismo de stop loss
df_pos = bt.create_backtest_returns(
    df_bt,
    leverage=bt_params['bt_params']['leverage'],
    stop_loss=10000 # coloco um stop de um retorno acumulado extremamente 
    # alto para não ser ativado
)

# Com mecanismo de stop loss
df_pos2 = bt.create_backtest_returns(
    df_bt,
    leverage=bt_params['bt_params']['leverage'],
    stop_loss=5/100
)

# Gráfico 4 - Figura 4 Retorno acumulado da estratégia de par com e 

fig, ax = plt.subplots(figsize=(10, 3.5), dpi=300)

df_rets = pd.concat([df_pos['strategy_acum'], df_pos2['strategy_acum']], axis=1)
df_rets.columns = ['Sem Stop', 'Com Stop']

ax.plot(df_rets['Sem Stop'], color='#979dac', label='Sem Stop')
ax.plot(df_rets['Com Stop'], color='#0466c8', label='Com Stop')
ax.axhline(y=-0.05, color='r', linestyle='--', label='Stop Loss')
plt.xlabel('Data')
ax.xaxis.set_major_formatter(
    mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
ax.grid(axis='x', linewidth=0.5)
ax.legend()
plt.show()

# plt.savefig('./charts/fig4.jpeg',  dpi=300)