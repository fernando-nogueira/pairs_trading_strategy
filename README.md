### Análise da estratégia de pairs trading por cointegração no mercado de ações brasileiro
Algoritmo desenvolvido na linguagem de programação Python utilizado na monografia para a estratégia modelo de pairs trading e suas simulações. 

### Módulos auxiliares
* Módulo para criar os sinais de entrada e saída das operações e acruar seus resultados ./src/backtest.py
* Módulo para ler o conjunto de dados: (./src/data.py)
* Módulo para criar a combinação de pares (./src/pairs.py)
* Módulo que cria os conjuntos de dados de forma rolante para separar teste de treinamento ./src/wfa.py
* Módulo com o procedimento de Johansen ./src/johansen_cointegration.py

### Módulos principais
* Módulo para criar uma estratégia: (./strategy.py)
* Módulo para criar simulações das estratégias: (./strat_simulator.py)
* Gráficos presentes no capítulo 2 de metodologia: (./charts_cp2.py)
* Resultados das simulações, gráficos e tabelas presentes no capítulo 3 de resultados empíricos (./results.py)
