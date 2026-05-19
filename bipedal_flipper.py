import gymnasium as gym
import numpy as np
import time

class BipedalFlipperWrapper(gym.Wrapper):
    def __init__(self, env, max_stagnation_steps=250):
        super().__init__(env)
        self.cumulative_angle = 0.0
        self.prev_angle = 0.0
        self.flip_completed = False
        
        # Stagnation tracking (250 steps = ~5 seconds of simulation time)
        self.max_stagnation_steps = max_stagnation_steps
        self.step_counter = 0
        self.last_x = 0.0
        self.last_progress_step = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.cumulative_angle = 0.0
        self.prev_angle = obs[0]  # obs[0] is the hull angle in BipedalWalker
        self.flip_completed = False
        
        self.step_counter = 0
        self.last_progress_step = 0
        try:
            self.last_x = self.env.unwrapped.hull.position.x
        except AttributeError:
            self.last_x = 0.0  # Fallback if hull isn't initialized yet
            
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        self.step_counter += 1
        
        # Check if doing any forward progress
        try:
            current_x = self.env.unwrapped.hull.position.x
            # If agent has advanced by at least 1 unit forward
            if current_x > self.last_x + 1.0:
                self.last_x = current_x
                self.last_progress_step = self.step_counter
        except AttributeError:
            pass
            
        # Terminate if stuck in the same spot for max_stagnation_steps
        stagnation_duration = self.step_counter - self.last_progress_step
        if stagnation_duration >= self.max_stagnation_steps:
            terminated = True
            
        current_angle = obs[0]
        
        # Calculate angle delta (handling potential wrapping)
        delta_angle = current_angle - self.prev_angle
        # Adjust for wrapping between -pi and pi
        if delta_angle > np.pi: 
            delta_angle -= 2 * np.pi
        elif delta_angle < -np.pi: 
            delta_angle += 2 * np.pi
            
        self.cumulative_angle += delta_angle
        self.prev_angle = current_angle

        # 1. Custom Reward: Reward for rotating backwards (backflip)
        # We replace the default forward-movement reward with a rotation reward
        custom_reward = -delta_angle * 10.0  # Positive reward for rotating backward
        
        # 2. Custom Condition: Did it complete a backflip? (-2pi radians)
        if self.cumulative_angle <= -2 * np.pi and not self.flip_completed:
            custom_reward += 100.0  # Big bonus for completing the flip
            self.flip_completed = True
            terminated = True # End the episode successfully
            
        # 3. Handle falling: The base env sets reward to -100 if it falls.
        if reward == -100: 
            if self.flip_completed:
                custom_reward = 0  # Ignore fall penalty if flip was done
            else:
                custom_reward -= 50 # Lighter penalty while learning

        # Optional: penalyze large motor torques
        custom_reward -= sum(abs(action)) * 0.01

        return obs, custom_reward, terminated, truncated, info

if __name__ == "__main__":
    import os
    from stable_baselines3 import SAC
    from stable_baselines3.common.vec_env import SubprocVecEnv
    from stable_baselines3.common.callbacks import BaseCallback

    class RenderCallback(BaseCallback):
        """Custom callback to occasionally pop open a window and show how the agent is doing."""
        def __init__(self, render_freq: int, verbose=0):
            super().__init__(verbose)
            self.render_freq = render_freq
            self.render_env = None

        def _on_step(self) -> bool:
            if self.num_timesteps > 0 and self.num_timesteps % self.render_freq == 0:
                print(f"\n--- [Step {self.num_timesteps}] Pausing parallel training to demonstrate current policy ---")
                if self.render_env is None:
                    # We create a single standard environment with human rendering enabled
                    e = gym.make("BipedalWalker-v3", hardcore=False, render_mode="human")
                    self.render_env = BipedalFlipperWrapper(e)
                
                obs, _ = self.render_env.reset()
                done = False
                while not done:
                    # Always use deterministic=True for evaluation
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, _, terminated, truncated, _ = self.render_env.step(action)
                    done = terminated or truncated
                
                print("--- Demonstration over. Resuming high-speed training ---\n")
            return True

    print("Setting up vectorized environments for training...")
    
    # Factory function to create isolated instances of our custom environment
    def make_env():
        def _init():
            # Render mode is omitted to allow fast simulation during training
            e = gym.make("BipedalWalker-v3", hardcore=False)
            e = BipedalFlipperWrapper(e)
            return e
        return _init

    # Number of parallel environments (matches the CPU cores for better data collection)
    num_envs = 8
    print(f"Creating {num_envs} parallel environments...")
    
    # SubprocVecEnv runs environments in separate CPU processes for true parallelism
    vec_env = SubprocVecEnv([make_env() for _ in range(num_envs)])
    
    model_path = "sac_bipedal_flipper.zip"
    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path} to continue training...")
        # Load the model and pass the environment so it can continue training
        model = SAC.load(model_path, env=vec_env, device="cuda")
    else:
        # Initialize a new Soft Actor-Critic (SAC) model, explicitly targeting the GPU
        print("Initializing new SAC model on GPU...")
        model = SAC("MlpPolicy", vec_env, verbose=1, device="cuda")
    
    # Train for a specified number of steps (increased since we gather data much faster now)
    total_steps = 20000
    print(f"Starting training for {total_steps} steps...")
    
    # Create the callback to show the agent on-screen every 50,000 steps
    # We only visualize periodically because rendering every step destroys performance. 
    eval_callback = RenderCallback(render_freq=1000)
    
    # Stable Baselines 3 supports a built-in progress bar (requires rich/tqdm)
    model.learn(total_timesteps=total_steps, callback=eval_callback, progress_bar=True)
    
    print("Training finished! Saving model...")
    model.save("sac_bipedal_flipper")
    
    vec_env.close()

    #(Optional) Uncomment below to see the trained agent in action:
    #print("Testing trained agent...")
    #test_env = gym.make("BipedalWalker-v3", hardcore=False, render_mode="human")
    #test_env = BipedalFlipperWrapper(test_env)
    #obs, info = test_env.reset()
    #while True:
    #    action, _states = model.predict(obs, deterministic=True)
    #    obs, reward, terminated, truncated, info = env.step(action)
    #     if terminated or truncated:
    #        obs, info = env.reset()
