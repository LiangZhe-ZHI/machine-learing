#ave_hi_nyc_jan_1895-2018
import pandas as pd
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt

nyc = pd.read_csv('resources/ave_hi_nyc_jan_1895-2018.csv')
nyc.columns = ['Date', 'Temperature', 'Anomaly']
nyc.Date = nyc.Date.floordiv(100)
pd.set_option('precision', 2)
# print(nyc.Temperature.describe())

linear_regression = stats.linregress(x=nyc.Date, y=nyc.Temperature)
# print(linear_regression.slope)
# print(linear_regression.intercept)

sns.set_style('whitegrid')
axes = sns.regplot(x=nyc.Date, y=nyc.Temperature)
axes.set_ylim(10, 70)
plt.savefig('resources/Temperature.png')
plt.show()
