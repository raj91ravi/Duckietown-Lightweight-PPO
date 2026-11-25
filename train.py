"""Reference implementation for legacy gym-duckietown + SB3 2.x."""
import argparse
from pathlib import Path
import numpy as np
import cv2
import gymnasium as gym
import torch as th
from torch import nn
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

class LaneEnv(gym.Env):
    def __init__(self, map_name='loop_empty', seed=42):
        from gym_duckietown.envs import DuckietownEnv
        self.sim = DuckietownEnv(map_name=map_name, seed=seed, domain_rand=True)
        self.observation_space = gym.spaces.Box(0,255,(84,84,3),np.uint8)
        # normalized forward speed and signed steering, not calibrated SI units
        self.action_space = gym.spaces.Box(np.array([0.,-1.],np.float32),np.array([1.,1.],np.float32))
        self.previous_steer=0.; self.steps=0
    def image(self, obs):
        return cv2.resize(np.asarray(obs), (84,84), interpolation=cv2.INTER_AREA).astype(np.uint8)
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None: self.sim.seed(seed)
        obs=self.sim.reset()
        if isinstance(obs,tuple): obs=obs[0]
        self.previous_steer=0.; self.steps=0
        return self.image(obs), {}
    def step(self, action):
        v, steer=map(float,action)
        result=self.sim.step(np.array([v,steer],np.float32))
        if len(result)==4:
            obs, _, done, info=result
            truncated=bool(info.get('TimeLimit.truncated',False)); terminated=bool(done and not truncated)
        else:
            obs, _, terminated, truncated, info=result
        self.steps+=1
        # Simulator ground truth is used only for training reward, not policy input.
        try:
            lane=self.sim.get_lane_pos2(self.sim.cur_pos,self.sim.cur_angle)
            lateral=float(lane.dist); heading=float(lane.angle_rad)
            if not np.isfinite([lateral,heading]).all(): raise ValueError('invalid lane pose')
            lane_reward=np.exp(-abs(lateral)/0.10)
            heading_reward=np.cos(heading)
            penalty=0.1*abs(steer-self.previous_steer)
            reward=1.0*lane_reward+0.5*heading_reward+0.2*v-penalty
        except Exception as exc:
            # Only known simulator off-lane exceptions are treated as termination.
            if type(exc).__name__ not in ('NotInLane','NotInLaneException'): raise
            reward=-5.; terminated=True
        if terminated: reward-=5.
        self.previous_steer=steer
        truncated=bool(truncated or (self.steps>=1000 and not terminated))
        return self.image(obs),float(reward),bool(terminated),truncated,info
    def close(self): self.sim.close()

class LightweightCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128):
        super().__init__(observation_space,features_dim)
        self.cnn=nn.Sequential(nn.Conv2d(3,32,8,4),nn.ReLU(),nn.Conv2d(32,64,4,2),nn.ReLU(),nn.Conv2d(64,64,3,1),nn.ReLU(),nn.Flatten())
        with th.no_grad(): n=self.cnn(th.zeros(1,3,84,84)).shape[1]
        self.fc=nn.Sequential(nn.Linear(n,features_dim),nn.ReLU())
    def forward(self, observations): return self.fc(self.cnn(observations))

class StopAtBudget(BaseCallback):
    # SB3 normally rounds the budget up to a full rollout; use its standard behavior.
    def _on_step(self): return True

def main():
    p=argparse.ArgumentParser(); p.add_argument('--steps',type=int,default=1_000_000); p.add_argument('--seed',type=int,default=42); p.add_argument('--map',default='loop_empty'); p.add_argument('--out',default='runs/ppo'); args=p.parse_args()
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    env=Monitor(LaneEnv(args.map,args.seed),str(out/'monitor.csv'))
    try:
        model=PPO('CnnPolicy',env,learning_rate=3e-4,gamma=.99,gae_lambda=.95,clip_range=.2,batch_size=64,n_steps=2048,n_epochs=10,policy_kwargs=dict(features_extractor_class=LightweightCNN,features_extractor_kwargs=dict(features_dim=128),net_arch=[],normalize_images=True),seed=args.seed,verbose=1,tensorboard_log=str(out/'tb'))
        model.learn(total_timesteps=args.steps)
        model.save(str(out/'policy'))
        print('Actual environment interactions:',model.num_timesteps)
    finally: env.close()
if __name__=='__main__': main()
