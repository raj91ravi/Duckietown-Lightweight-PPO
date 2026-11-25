import argparse
import pandas as pd
import matplotlib.pyplot as plt
p=argparse.ArgumentParser(); p.add_argument('monitor'); p.add_argument('--window',type=int,default=20); a=p.parse_args()
d=pd.read_csv(a.monitor,comment='#'); x=d.l.cumsum(); mean=d.r.rolling(a.window,min_periods=1).mean(); sd=d.r.rolling(a.window,min_periods=2).std().fillna(0)
plt.plot(x,mean,label='Rolling mean episodic reward'); plt.fill_between(x.to_numpy(),(mean-sd).to_numpy(),(mean+sd).to_numpy(),alpha=.2,label='Rolling ±1 SD'); plt.xlabel('Environment interaction steps (completed episodes)'); plt.ylabel('Episode reward'); plt.legend(); plt.tight_layout(); plt.savefig('training_curve.pdf'); plt.show()
