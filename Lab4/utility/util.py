from matplotlib import pyplot as plt
import warnings
import seaborn as sns


# Plotting Support Functions
def configure_plots():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        for _ in range(2): # needs to run twice for some reason
            sns.set(style="ticks", color_codes=True, font_scale=1.5)
            sns.set_palette(sns.color_palette())
            plt.rcParams['figure.figsize'] = [16, 9]
            plt.rcParams['axes.titlesize'] = 20
            plt.rcParams['axes.labelsize'] = 16
            plt.rcParams['xtick.labelsize'] = 14
            plt.rcParams['ytick.labelsize'] = 14
            plt.rcParams['lines.linewidth'] = 2

        print('Plots configured! 📊')