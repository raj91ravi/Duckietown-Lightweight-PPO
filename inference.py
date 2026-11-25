"""Offline inference only: no motor commands are transmitted."""
import argparse, cv2, numpy as np, time
from stable_baselines3 import PPO
from train import LightweightCNN

def predict(model, rgb):
    obs=cv2.resize(rgb,(84,84),interpolation=cv2.INTER_AREA).astype(np.uint8)
    action,_=model.predict(obs,deterministic=True)
    return float(action[0]),float(action[1])

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('model'); p.add_argument('image'); a=p.parse_args()
    model=PPO.load(a.model,device='cpu'); bgr=cv2.imread(a.image)
    if bgr is None: raise ValueError('Cannot read image')
    rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
    t=time.perf_counter(); print('Normalized speed, steering:',predict(model,rgb)); print('Preprocessing + prediction ms:',(time.perf_counter()-t)*1000)
