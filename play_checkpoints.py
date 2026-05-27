import os
import glob
import time
import gymnasium as gym
import numpy as np
import custom_bipedal
from stable_baselines3 import PPO

# ==============================================================================
# WRAPPERS
# ==============================================================================

class CurriculumFlipperWrapper(gym.Wrapper):
    def __init__(self, env, stage=1, max_steps=400):
        super().__init__(env)
        self.max_steps = max_steps
        self.stage = stage 
        
        self.cumulative_angle = 0.0
        self.prev_angle = 0.0
        self.flip_completed = False
        self.step_counter = 0
        
        low = np.append(self.env.observation_space.low, -np.inf)
        high = np.append(self.env.observation_space.high, np.inf)
        self.observation_space = gym.spaces.Box(low, high, dtype=np.float32)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.step_counter += 1

        current_angle = obs[0]
        delta_angle = current_angle - self.prev_angle

        if delta_angle > np.pi:
            delta_angle -= 2 * np.pi
        elif delta_angle < -np.pi:
            delta_angle += 2 * np.pi

        prev_cumulative = self.cumulative_angle
        self.cumulative_angle += delta_angle
        self.prev_angle = current_angle

        abs_angle = abs(self.cumulative_angle)
        
        if abs_angle >= 5.2 and not self.flip_completed:
            self.flip_completed = True

        if self.step_counter >= self.max_steps:
            truncated = True

        info['flip_completed'] = self.flip_completed
        info['cumulative_angle'] = self.cumulative_angle

        # Feed the absolute angle percentage to the observation network
        obs = np.append(obs, abs_angle / (2 * np.pi)).astype(np.float32)
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.cumulative_angle = 0.0
        self.prev_angle = obs[0]
        self.flip_completed = False
        self.step_counter = 0
        obs = np.append(obs, 0.0).astype(np.float32)
        return obs, info

class Stage3BalanceWrapper(CurriculumFlipperWrapper):
    def __init__(self, env, max_steps=400):
        super().__init__(env, stage=2, max_steps=max_steps)
        self.landed_steps = 0
        self.in_balance_phase = False

    def step(self, action):
        if not self.in_balance_phase:
            obs, reward, terminated, truncated, info = super().step(action)
            both_feet = (obs[8] == 1.0 and obs[13] == 1.0)
            if self.flip_completed and both_feet:
                self.in_balance_phase = True
            return obs, reward, terminated, truncated, info
        else:
            raw_obs, raw_reward, terminated, truncated, raw_info = self.env.step(action)
            self.step_counter += 1

            current_angle = raw_obs[0]
            delta_angle = current_angle - self.prev_angle
            if delta_angle > np.pi:    delta_angle -= 2 * np.pi
            elif delta_angle < -np.pi: delta_angle += 2 * np.pi
            self.cumulative_angle += delta_angle
            self.prev_angle = current_angle
            abs_angle = abs(self.cumulative_angle)

            both_feet = (raw_obs[8] == 1.0 and raw_obs[13] == 1.0)
            self.landed_steps += 1 if both_feet else 0

            if self.step_counter >= self.max_steps:
                truncated = True

            raw_info['flip_completed'] = self.flip_completed
            raw_info['cumulative_angle'] = self.cumulative_angle

            final_obs = np.append(raw_obs, abs_angle / (2 * np.pi)).astype(np.float32)
            return final_obs, raw_reward, terminated, truncated, raw_info

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self.landed_steps = 0
        self.in_balance_phase = False
        obs_array = obs[:-1]
        abs_angle = abs(self.cumulative_angle)
        return np.append(obs_array, abs_angle / (2 * np.pi)).astype(np.float32), info

class Stage4ChainFlipWrapper(Stage3BalanceWrapper):
    def __init__(self, env, max_steps=400):
        super().__init__(env, max_steps=max_steps)
        self.flip_count = 0
        self.prev_flip_completed = False
        self.chain_phase = False
        self.chain_steps = 0
        self.last_flip_angle = 0.0

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        flips_done = int(abs(self.cumulative_angle) / (2 * np.pi))
        
        if flips_done > self.flip_count:
            self.flip_count = flips_done
            self.chain_phase = True
            self.chain_steps = 0
            self.last_flip_angle = abs(self.cumulative_angle)

        if self.chain_phase:
            self.chain_steps += 1
            both_feet = (obs[8] == 1.0 and obs[13] == 1.0)
            is_airborne = (obs[8] == 0.0 and obs[13] == 0.0)

            if self.chain_steps > 400 and not is_airborne:
                truncated = True
                
        info['flip_count'] = self.flip_count
        info['chain_phase'] = self.chain_phase
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self.flip_count = 0
        self.prev_flip_completed = False
        self.chain_phase = False
        self.chain_steps = 0
        self.last_flip_angle = 0.0
        return obs, info

def get_wrapper_for_stage(env, stage_num):
    if stage_num == 1 or stage_num == 2:
        return CurriculumFlipperWrapper(env, stage=stage_num)
    elif stage_num == 3:
        return Stage3BalanceWrapper(env)
    elif stage_num == 4:
        return Stage4ChainFlipWrapper(env)
    return env

# ==============================================================================
# MAIN DISPLAY LOOP
# ==============================================================================

def play_checkpoints():
    checkpoints_dir = "checkpoints"
    if not os.path.exists(checkpoints_dir):
        print("Checkpoints directory not found!")
        return

    stages = [1, 2, 3, 4]
    
    for stage_num in stages:
        stage_dir = os.path.join(checkpoints_dir, f"stage{stage_num}")
        if not os.path.exists(stage_dir):
            continue
            
        print(f"\n" + "="*50)
        print(f"🎬 PREVIEWING STAGE {stage_num}")
        print("="*50)

        # Gather zip files and sort by step count
        checkpoints = glob.glob(os.path.join(stage_dir, "*.zip"))
        
        def get_step_count(filepath):
            filename = os.path.basename(filepath)
            parts = filename.replace('.zip', '').split('_')
            for part in parts:
                if part.isdigit():
                    return int(part)
            return 0
            
        checkpoints.sort(key=get_step_count)
        
        if not checkpoints:
            print(f"No checkpoint zip files found in {stage_dir}")
            continue

        for ckpt in checkpoints:
            print(f"\n=> Loading checkpoint: {os.path.basename(ckpt)}")
            
            try:
                model = PPO.load(ckpt, device="cpu")
            except Exception as e:
                print(f"Error loading {ckpt}: {e}")
                continue

            env = custom_bipedal.BipedalWalker(hardcore=False, render_mode="human")
            env = get_wrapper_for_stage(env, stage_num)
            
            obs, info = env.reset()
            done = False
            total_reward = 0.0
            step_count = 0
            
            # Display until the episode finishes (or max 2000 steps to prevent infinite hangs)
            while not done and step_count < 2000:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                total_reward += reward
                step_count += 1
                
                time.sleep(0.01)  # small delay for smoother viewing
            
            print(f"   Episode finished after {step_count} steps. Total Reward: {total_reward:.2f}")
            env.close()

if __name__ == '__main__':
    play_checkpoints()