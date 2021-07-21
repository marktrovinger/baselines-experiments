import gym
from stable_baselines3 import A2C
from stable_baselines3 import DQN
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from wandb.integration.sb3 import WandbCallback
import wandb

# since we are playing breakout, we need a CNN policy
config = {"policy_type": "CnnPolicy",
          "total_timesteps": 25000,
          "env_name":"Breakout-v0"}

def make_env():
    env = gym.make(config.get('env_name'))
    env = Monitor(env)  # record stats such as returns
    return env

env = DummyVecEnv([make_env])

env = VecVideoRecorder(env, "videos",
    record_video_trigger=lambda x: x % 2000 == 0, video_length=200)

run = wandb.init(
    project="sb3",
    config=config,
    sync_tensorboard=True,
    monitor_gym=True,
    save_code=True
)

model = DQN(config["policy_type"], env, verbose=1, 
            tensorboard_log=f"runs/{run.id}")
model.learn(total_timesteps=config["total_timesteps"],
            callback=WandbCallback(
                gradient_save_freq=100,
                model_save_path=f"models/{run.id}"
            ),)

