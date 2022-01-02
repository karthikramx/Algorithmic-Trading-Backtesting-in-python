=======
# Algorithmic Trading (Backtesting in python)
Code snippet show casing backtesting a strategy in python on historical stock market data.


<font size=3><strong>Intro</strong></font><br />
The goal of this article is to describe how to back-test a technical indicator-based strategy on python. I will specifically use a Bollinger band-based strategy to create signals and positions.

<font size=3><strong>Description of strategy</strong></font><br />
Create 20-day (+/- 2 standard deviations) Bollinger bands on the adjusted close price. Buy, when the price crosses the lower band from the top and hold until the price crosses the upper band from below the next time. Sell when the price crosses the upper band from below and hold until the price crosses the lower band from the top the next time.

<font size=3><strong>Here are the steps to create your own back-testing code.</strong></font> <br />

1. Import necessary libraries
2. Download OHLCV Data
3. Calculate daily returns
4. Create strategy-based data columns
5. Create strategy indicators
6. Create signals and positions
7. Analyze results

<font size=3><strong>Step 1 : Import necessary libraries</strong></font>


```python
# Ignore printing all warnings
warnings.filterwarnings('ignore')

# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import pyfolio as pf
import datetime as dt
import pandas_datareader.data as web
import os
import warnings

# print all outputs
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
```

<font size=3><strong>Step 2 : Download OHLCV: (Open, High, Low, Close, Volume) data</strong></font></br>
I use yahoo finance python API - yfinance to get the data. There are a lot of resources to get historical data in order to backtest your strategies.


```python
# downloading historical necessary data for backtesting and analysis
_start = dt.date(2015,1,2)
_end = dt.date(2020,4,30)
ticker = 'MSFT'
df = yf.download(ticker, start = _start, end = _end) 
```

    [*********************100%***********************]  1 of 1 completed


<font size=3><strong>Step 3 : Calculate daily returns</strong></font></br>
This step calculates daily returns for comparing performance with the buy and hold strategy. A buy and hold strategy becomes a benchmark or comparing the strategy. In other words, it checks if the strategy performed better than simply buying and holding the stock. A good strategy would essentially perform better than a buy-and-hold strategy.


```python
# calculating buy and hold strategy returns
df['bnh_returns'] = np.log(df['Adj Close']/df['Adj Close'].shift(1))
df.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
      <th>Close</th>
      <th>Adj Close</th>
      <th>Volume</th>
      <th>bnh_returns</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2015-01-02</th>
      <td>46.660000</td>
      <td>47.419998</td>
      <td>46.540001</td>
      <td>46.759998</td>
      <td>41.193840</td>
      <td>27913900</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2015-01-05</th>
      <td>46.369999</td>
      <td>46.730000</td>
      <td>46.250000</td>
      <td>46.330002</td>
      <td>40.815018</td>
      <td>39673900</td>
      <td>-0.009239</td>
    </tr>
    <tr>
      <th>2015-01-06</th>
      <td>46.380001</td>
      <td>46.750000</td>
      <td>45.540001</td>
      <td>45.650002</td>
      <td>40.215977</td>
      <td>36447900</td>
      <td>-0.014786</td>
    </tr>
  </tbody>
</table>
</div>



<font size=3><strong>Step 4 : Create strategy-based data columns</strong></font></br>
The next step is to create indicators to generate conditions of the strategy. For Bollinger band strategy, involves the 20-day moving average, the standard deviation of the 20 days moving average, upper band, and lower band of the standard deviation. [ma20,std,upper_band,lower_band]


```python
# creating bollinger band indicators
df['ma20'] = df['Adj Close'].rolling(window=20).mean()
df['std'] = df['Adj Close'].rolling(window=20).std()
df['upper_band'] = df['ma20'] + (2 * df['std'])
df['lower_band'] = df['ma20'] - (2 * df['std'])
df.drop(['Open','High','Low'],axis=1,inplace=True,errors='ignore')
df.tail(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Close</th>
      <th>Adj Close</th>
      <th>Volume</th>
      <th>bnh_returns</th>
      <th>ma20</th>
      <th>std</th>
      <th>upper_band</th>
      <th>lower_band</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-04-23</th>
      <td>171.419998</td>
      <td>168.672516</td>
      <td>32790800</td>
      <td>-0.012176</td>
      <td>162.283099</td>
      <td>8.605524</td>
      <td>179.494147</td>
      <td>145.072050</td>
    </tr>
    <tr>
      <th>2020-04-24</th>
      <td>174.550003</td>
      <td>171.752350</td>
      <td>34277600</td>
      <td>0.018095</td>
      <td>163.190320</td>
      <td>8.599244</td>
      <td>180.388808</td>
      <td>145.991831</td>
    </tr>
    <tr>
      <th>2020-04-27</th>
      <td>174.050003</td>
      <td>171.260361</td>
      <td>33194400</td>
      <td>-0.002869</td>
      <td>164.388305</td>
      <td>7.910466</td>
      <td>180.209236</td>
      <td>148.567373</td>
    </tr>
    <tr>
      <th>2020-04-28</th>
      <td>169.809998</td>
      <td>167.088333</td>
      <td>34392700</td>
      <td>-0.024662</td>
      <td>164.859628</td>
      <td>7.768141</td>
      <td>180.395909</td>
      <td>149.323347</td>
    </tr>
    <tr>
      <th>2020-04-29</th>
      <td>177.429993</td>
      <td>174.586166</td>
      <td>51286600</td>
      <td>0.043896</td>
      <td>165.829823</td>
      <td>7.707362</td>
      <td>181.244546</td>
      <td>150.415099</td>
    </tr>
  </tbody>
</table>
</div>



<font size=3><strong>Step 5 : Create strategy indicators</strong></font></br>
The following is the most crucial part of creating the strategy. It involves
- generating long and short signals as mentioned in the strategy
- replacing zeros with forwarding fill, to generate long and short positions
- shifting positions by 1 to signify return calculations done from close of a day to the next day close price

Signals are essentially indicators that signify the action that needs to be taken (ie: to buy or sell). Positions are what you maintain after buying or selling (ie: going long or short).


```python
# BUY condition
df['signal'] = np.where( (df['Adj Close'] < df['lower_band']) &
                          (df['Adj Close'].shift(1) >= df['lower_band']),1,0)

# SELL condition
df['signal'] = np.where( (df['Adj Close'] > df['upper_band']) &
                          (df['Adj Close'].shift(1) <= df['upper_band']),-1,df['signal'])
# creating long and short positions 
df['position'] = df['signal'].replace(to_replace=0, method='ffill')

# shifting by 1, to account of close price return calculations
df['position'] = df['position'].shift(1)

# calculating stretegy returns
df['strategy_returns'] = df['bnh_returns'] * (df['position'])

df.tail(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Close</th>
      <th>Adj Close</th>
      <th>Volume</th>
      <th>bnh_returns</th>
      <th>ma20</th>
      <th>std</th>
      <th>upper_band</th>
      <th>lower_band</th>
      <th>signal</th>
      <th>position</th>
      <th>strategy_returns</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-04-23</th>
      <td>171.419998</td>
      <td>168.672516</td>
      <td>32790800</td>
      <td>-0.012176</td>
      <td>162.283099</td>
      <td>8.605524</td>
      <td>179.494147</td>
      <td>145.072050</td>
      <td>0</td>
      <td>1.0</td>
      <td>-0.012176</td>
    </tr>
    <tr>
      <th>2020-04-24</th>
      <td>174.550003</td>
      <td>171.752350</td>
      <td>34277600</td>
      <td>0.018095</td>
      <td>163.190320</td>
      <td>8.599244</td>
      <td>180.388808</td>
      <td>145.991831</td>
      <td>0</td>
      <td>1.0</td>
      <td>0.018095</td>
    </tr>
    <tr>
      <th>2020-04-27</th>
      <td>174.050003</td>
      <td>171.260361</td>
      <td>33194400</td>
      <td>-0.002869</td>
      <td>164.388305</td>
      <td>7.910466</td>
      <td>180.209236</td>
      <td>148.567373</td>
      <td>0</td>
      <td>1.0</td>
      <td>-0.002869</td>
    </tr>
    <tr>
      <th>2020-04-28</th>
      <td>169.809998</td>
      <td>167.088333</td>
      <td>34392700</td>
      <td>-0.024662</td>
      <td>164.859628</td>
      <td>7.768141</td>
      <td>180.395909</td>
      <td>149.323347</td>
      <td>0</td>
      <td>1.0</td>
      <td>-0.024662</td>
    </tr>
    <tr>
      <th>2020-04-29</th>
      <td>177.429993</td>
      <td>174.586166</td>
      <td>51286600</td>
      <td>0.043896</td>
      <td>165.829823</td>
      <td>7.707362</td>
      <td>181.244546</td>
      <td>150.415099</td>
      <td>0</td>
      <td>1.0</td>
      <td>0.043896</td>
    </tr>
  </tbody>
</table>
</div>



<font size=3><strong>Step 6 : Create signals and positions</strong></font></br>
The next step is to compare the strategy performance using cumulative returns. 
 This involves element-wise multiplication of the positions with the daily returns.


```python
# comparing buy & hold strategy / bollinger bands strategy returns
print("Buy and hold returns:",df['bnh_returns'].cumsum()[-1])
print("Strategy returns:",df['strategy_returns'].cumsum()[-1])

# plotting strategy historical performance over time
df[['bnh_returns','strategy_returns']] = df[['bnh_returns','strategy_returns']].cumsum()
df[['bnh_returns','strategy_returns']].plot(grid=True, figsize=(12, 8))
```

    Buy and hold returns: 1.4441296786561466
    Strategy returns: 0.37122507466278376





    <AxesSubplot:xlabel='Date'>




    
![png](output_12_2.png)
    


<font size=3><strong>Step 7 : Analyze results</strong></font></br>

For this step I use pyfolio. Pyfolio is a Python library for performance and risk analysis of financial portfolios


```python
pf.create_simple_tear_sheet(df['strategy_returns'].diff())
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;"><th>Start date</th><td colspan=2>2015-01-02</td></tr>
    <tr style="text-align: right;"><th>End date</th><td colspan=2>2020-04-29</td></tr>
    <tr style="text-align: right;"><th>Total months</th><td colspan=2>63</td></tr>
    <tr style="text-align: right;">
      <th></th>
      <th>Backtest</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Annual return</th>
      <td>3.3%</td>
    </tr>
    <tr>
      <th>Cumulative returns</th>
      <td>19.1%</td>
    </tr>
    <tr>
      <th>Annual volatility</th>
      <td>27.1%</td>
    </tr>
    <tr>
      <th>Sharpe ratio</th>
      <td>0.26</td>
    </tr>
    <tr>
      <th>Calmar ratio</th>
      <td>0.12</td>
    </tr>
    <tr>
      <th>Stability</th>
      <td>0.16</td>
    </tr>
    <tr>
      <th>Max drawdown</th>
      <td>-28.6%</td>
    </tr>
    <tr>
      <th>Omega ratio</th>
      <td>1.05</td>
    </tr>
    <tr>
      <th>Sortino ratio</th>
      <td>0.37</td>
    </tr>
    <tr>
      <th>Skew</th>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Kurtosis</th>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Tail ratio</th>
      <td>1.16</td>
    </tr>
    <tr>
      <th>Daily value at risk</th>
      <td>-3.4%</td>
    </tr>
  </tbody>
</table>



    
![png](output_14_1.png)
    


<font size=3><strong>Comments</strong></font></br>
The results from pyfolio are self-explanatory.

The 'Bollinger band' strategy on the historical data did not perform better than the buy & hold strategy. The cumulative daily returns for buy and hold accounts to 1.44 times the initial investment and the Bollinger band strategy returns account for 0.37 times the initial investment.

For obvious reasons, one should not take a strategy live, even if it gives great returns on back-testing. There are various risks involved such as not accounting for transaction costs and momentum in stock price.

The performance of a strategy can also be optimized by checking the returns on various strategy parameters.

Various other technical indicators can be used in conjunction to create signals which can reduce risks and improve performance. Live strategies should implement stop loss and kill switches in case the strategy goes of hand.

This code snippet can be found at : https://github.com/karthikramx/Algorithmic-Trading-Backtesting-python-example

