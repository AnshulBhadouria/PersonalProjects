import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib as mpl
import matplotlib.pyplot as plt   # data visualization
import seaborn as sns             # statistical data visualization

filepath = r"C:\Users\ana18\OneDrive\Desktop\Python\myvenv\Practice\Projects\ts_forecasting\AirPassengers.csv"
df = pd.read_csv(filepath)
print(df.head(10))

df.columns = ['Date','Number of Passengers']

def plot_df(df, x, y, title="", xlabel='Date', ylabel='Number of Passengers', dpi=100):
    plt.figure(figsize=(15,4), dpi=dpi)
    plt.plot(x, y, color='tab:red')
    plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)
    plt.show()
    

