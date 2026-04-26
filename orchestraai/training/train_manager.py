import os
import yaml
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import (
    CheckpointCallback, EvalCallback, CallbackList
)
import wandb
from wandb.integration.sb3 import WandbCallback
import argparse

from env.manager_worker_env import ManagerWorkerEnv

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume")
    args = parser.parse_args()

    # Load config
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    # Init W&B
    run = wandb.init(
        project="orchestraai",
        config=config,
        sync_tensorboard=True,
        monitor_gym=True,
        save_code=True,
    )

    # Create Envs
    env = ManagerWorkerEnv("config.yaml")
    eval_env = ManagerWorkerEnv("config.yaml")
    
    # Check env
    print("Checking environment compliance...")
    check_env(env)

    # Model
    if args.resume:
        print(f"Resuming training from {args.resume}")
        model = PPO.load(args.resume, env=env)
    else:
        model = PPO(
            "MultiInputPolicy",
            env,
            learning_rate=config['training']['learning_rate'],
            n_steps=config['training']['n_steps'],
            batch_size=config['training']['batch_size'],
            n_epochs=config['training']['n_epochs'],
            gamma=config['training']['gamma'],
            gae_lambda=config['training']['gae_lambda'],
            clip_range=config['training']['clip_range'],
            ent_coef=config['training']['ent_coef'],
            verbose=1,
            tensorboard_log=config['training']['log_dir']
        )

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=config['training']['save_freq'],
        save_path=config['training']['model_save_path'],
        name_prefix="ppo_manager"
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=config['training']['model_save_path'],
        log_path=config['training']['log_dir'],
        eval_freq=5000,
        deterministic=True,
        render=False
    )
    
    wandb_callback = WandbCallback(
        gradient_save_freq=100,
        model_save_path=f"models/{run.id}",
        verbose=2,
    )

    callback_list = CallbackList([checkpoint_callback, eval_callback, wandb_callback])

    # Train
    model.learn(
        total_timesteps=config['training']['total_timesteps'],
        callback=callback_list,
        progress_bar=True
    )

    # Save final model
    model.save(config['training']['model_save_path'] + "_final")
    print(f"Training complete. Model saved to {config['training']['model_save_path']}_final")

    run.finish()

if __name__ == "__main__":
    train()
