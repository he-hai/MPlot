# %% 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from lmfit.models import GaussianModel

plt.rcParams["font.family"] = "Arial"
plt.rcParams['ytick.labelsize'] = 15
plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['axes.titlesize'] = 24
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['legend.title_fontsize'] = 15
plt.rcParams['legend.fontsize'] = 13
plt.rcParams.update({'mathtext.default': 'regular'})

#%% 
df = pd.read_csv('eventsFitted_DEMO.csv',comment='#')
mw = df[df['masses_kDa']>0]['masses_kDa']

bin_width = 6
mw_max = 800
bins = np.arange(0,mw_max, bin_width)

counts = mw.value_counts(bins=bins)
x = counts.index.mid
y = counts.to_numpy()

#%%
gmod1 = GaussianModel(prefix='g1_')
pars=gmod1.make_params()
pars['g1_center'].set(value=94,min=70,max=130)
pars['g1_sigma'].set(value=10)
pars['g1_amplitude'].set(value=1000)

# gmod2 = GaussianModel(prefix='g2_')
# pars.update(gmod2.make_params())
# pars['g2_center'].set(value=47,min=30,max=70)
# pars['g2_sigma'].set(value=10)
# pars['g2_amplitude'].set(value=1000)

model = gmod1
# model = gmod1 + gmod2

out = model.fit(y,pars,x=x)

# print(out.fit_report())

comps=out.eval_components(x=x)

#%%
fig, ax = plt.subplots(figsize=(8,6))
sns.histplot(data=mw, ax=ax, binwidth=bin_width)
# sns.lineplot(x=x,y=out.best_fit,ax=ax,color='k')
sns.lineplot(
    x=x,y=comps['g1_'],ax=ax,color='r',ls='--',
    label=f"MW={out.values['g1_center']:.0f}, $\sigma$={out.values['g1_sigma']:.0f}"
)

# sns.lineplot(
#     x=x,y=comps['g2_'],ax=ax,color='g',ls='--',
#     label=f"MW={out.values['g2_center']:.0f}, $\sigma$={out.values['g2_sigma']:.0f}"
# )

plt.xlabel('Mass [kDa]')
plt.xlim((0,mw_max))
plt.ylabel('Counts')
plt.title('Demo')

# fig.savefig('Demo_MP.png',bbox='tight')
plt.show()
