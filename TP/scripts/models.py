
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_predict
from statsmodels.tsa.statespace import sarimax
import seaborn as sns
import matplotlib.pyplot as plt


# Creo una función para evaluar los modelos
def evaluate_model(series, my_order, my_seasonal_order):
    
    mod=sm.tsa.statespace.SARIMAX(series, order = my_order, seasonal_order = my_seasonal_order)
    
    result = mod.fit(disp=False)
    print(result.summary())
    
    return mod, result

def get_resid_info(residual, bins):
    # Normalize residual
    resid = residual
    resid_norm = (resid-resid.mean())/resid.std()

    print(f"RESIDUALS STATISTICS \n {resid.describe()}")

    #Plot
    fig, ax = plt.subplots(2,3,figsize=(15,10))
    resid.plot(ax=ax[0,0])
    ax[0,0].set_title('Residuals')
    sm.qqplot(resid_norm, scale=1, line="45", ax=ax[0,1])
    ax[0,1].set_title('QQ Plot of Normalized Residuals')
    sm.graphics.tsa.plot_acf(resid, ax=ax[0,2])
    sns.histplot(resid_norm, bins=bins, ax=ax[1,0])
    ax[1,0].set_title(f'Histogram {bins} bins (normalized Residuals)')
    sns.boxplot(resid_norm, ax=ax[1,1])
    ax[1,1].set_title('Normalized Residuals Box Plot')
    plt.delaxes(ax[1][2])
    plt.tight_layout()
    plt.show()
    return