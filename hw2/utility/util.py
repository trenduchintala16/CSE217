import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import warnings
import seaborn as sns


# Plotting Support Functions
def configure_plots():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sns.set(style="ticks", color_codes=True, font_scale=1.5)
        sns.set_palette(sns.color_palette())
        plt.rcParams['figure.figsize'] = [16, 9]

def clean_columns(df):
    """Remove spaces and parentheses in column names."""
    df.columns = [
        col.replace(' ', '_').replace("(", "").replace(")", "").replace(
            ".", "") for col in df.columns
    ]
    return df


def plot_gender_fraction_over_time(df, title=None):
    """Calculates the annual fraction of artworks by gender. Plots the ratio over time."""
    
    # Drop entries where date is not defined
    df = df[df['DateAcquired'].notnull()]
    
    # Sort by date
    df = df.sort_values(by='DateAcquired')
    df = df.set_index('DateAcquired')
    
    # Add artworks acquired
    df['ones'] = 1
    df['num_acquired'] = df.ones.cumsum()
    
    works_man = df[(df.num_males>0) & (df.num_females==0)].copy()   # works by male artist(s)
    works_man['num_acquired'] = works_man.ones.cumsum()
    
    works_woman = df[(df.num_females>0) & (df.num_males==0)].copy() # works by female artist(s)
    works_woman['num_acquired'] = works_woman.ones.cumsum()
    
    # Compute the fraction of artworks from male and female artists acquired by year
    # [OPTIONAL]: the following operations are pretty advanced - you may ignore them  
    time_group = 'Y'
    frac_male_per_year = (100 * (works_man.groupby(pd.Grouper(freq=time_group))['ones'].count()) /
         (df.groupby(pd.Grouper(freq=time_group))['ones'].count()))
    
    frac_female_per_year =(100 * (works_woman.groupby(pd.Grouper(freq=time_group))['ones'].count()) /
         (df.groupby(pd.Grouper(freq=time_group))['ones'].count()))
    
    # Plot (this is a line plot)
    year_range = np.array(range(1928,2018))
    plt.plot(year_range,frac_male_per_year, label='Male')
    plt.plot(year_range,frac_female_per_year, label='Female')
    
    # add horizontal line at 50%
    plt.plot(year_range,np.ones(year_range.shape)*50, lw=0.5, color='gray')

    plt.xlabel("Year Acquired")
    plt.ylabel("Annual Percentage of Works")
    if title:
        plt.title(title)
    plt.legend()
    plt.show()