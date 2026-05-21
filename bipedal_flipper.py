import gymnasium as gym
import numpy as np

# BipedalWalker-v3 Observation Space (24 dimensions):
# obs[0]  = hull angle (radians)
# obs[1]  = hull angular velocity
# obs[2]  = hull horizontal velocity
# obs[3]  = hull vertical velocity
# obs[4]  = leg1 joint1 angle
# obs[5]  = leg1 joint1 speed
# obs[6]  = leg1 joint2 angle
# obs[7]  = leg1 joint2 speed
# obs[8]  = leg1 ground contact (bool)
# obs[9]  = leg2 joint1 angle
# obs[10] = leg2 joint1 speed
# obs[11] = leg2 joint2 angle
# obs[12] = leg2 joint2 speed
# obs[13] = leg2 ground contact (bool)
# obs[14..23] = 10 lidar range-finder readings


class BipedalFlipperWrapper(gym.Wrapper):
    def __init__(self, env, max_stagnation_steps=400):
        super().__init__(env)
        self.cumulative_angle = 0.0
        self.prev_angle = 0.0
        self.flip_completed = False
        self.landed = False

        # Stagnation tracking (250 steps = ~5 seconds of simulation time)
        self.max_stagnation_steps = max_stagnation_steps
        self.step_counter = 0
        self.last_x = 0.0
        self.last_progress_step = 0

        # Per-episode metric tracking (exposed via info on episode end)
        self.episode_max_rotation = 0.0   # Most positive cumulative angle seen
        self.episode_max_height = 0.0     # Highest hull_y seen above baseline

        # Reward accumulators
        self.ep_reward_rotation = 0.0
        self.ep_reward_penalty = 0.0
        self.ep_reward_jump = 0.0
        self.ep_reward_legs = 0.0
        self.ep_reward_landing = 0.0
        self.ep_reward_ang_vel = 0.0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.cumulative_angle = 0.0
        self.prev_angle = obs[0]   # hull angle
        self.flip_completed = False
        self.landed = False

        self.step_counter = 0
        self.last_progress_step = 0
        try:
            self.last_x = self.env.unwrapped.hull.position.x
        except AttributeError:
            self.last_x = 0.0

        self.episode_max_rotation = 0.0
        self.episode_max_height = 0.0
        self.ep_reward_rotation = 0.0
        self.ep_reward_penalty = 0.0
        self.ep_reward_jump = 0.0
        self.ep_reward_legs = 0.0
        self.ep_reward_landing = 0.0
        self.ep_reward_ang_vel = 0.0

        return obs, info

    def step(self, action):
        obs, base_reward, terminated, truncated, info = self.env.step(action)

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

        # --- Angle tracking ---
        current_angle = obs[0]          # hull angle in radians

        delta_angle = current_angle - self.prev_angle
        # Wrap delta to [-pi, pi] to handle the -pi/+pi boundary
        if delta_angle > np.pi:
            delta_angle -= 2 * np.pi
        elif delta_angle < -np.pi:
            delta_angle += 2 * np.pi

        # In BipedalWalker, leaning forward decreases the angle (negative rotation).
        # We invert it here so that frontflips result in POSITIVE cumulative angle and rewards.
        delta_angle = -delta_angle

        self.cumulative_angle += delta_angle
        self.prev_angle = current_angle

        # Track per-episode max rotation (most positive = most forward rotation achieved)
        if self.cumulative_angle > self.episode_max_rotation:
            self.episode_max_rotation = self.cumulative_angle

        # 1. Custom Reward: Reward for rotating forward (frontflip)
        # We replace the default forward-movement reward with a rotation reward
        # Exponentially scale up the reward as it gets closer to a full flip (2*pi)
        progress_ratio = self.cumulative_angle / (2 * np.pi)
        
        # Base reward for rotating
        rot_rew = delta_angle * 10.0  
        
        # HUGE continuous shaping reward for getting closer to the goal
        if delta_angle > 0:
            # Multiplier increases from 1x to ~10x as it nears a full flip
            multiplier = 1.0 + (progress_ratio * 10.0)
            rot_rew *= multiplier

        custom_reward = rot_rew
        self.ep_reward_rotation += rot_rew

        # 2. Custom Condition: Did it complete a frontflip? (2pi radians)
        if self.cumulative_angle >= 2 * np.pi and not self.flip_completed:
            custom_reward += 500.0  # Massive bonus for completing the flip
            self.ep_reward_rotation += 500.0
            self.flip_completed = True
            # NOTE: We DO NOT terminate here anymore. We want it to land on its feet!

        # 3. Custom Condition: Reward landing safely after the flip
        if self.flip_completed and not self.landed:
            # If hull is roughly upright again (modulo 2pi, roughly 0.0 radians to ground)
            # AND at least one leg touches the ground
            if abs(obs[0]) < 0.5 and (obs[8] or obs[13]):
                custom_reward += 500.0
                self.ep_reward_landing += 500.0
                self.landed = True
                terminated = True  # Fully successful flip and landing!

        # 4. Handle falling: The base env sets reward to -20 if it falls.
        if base_reward == -20:
            custom_reward -= 20.0  # Heavy penalty for falling, whether before or after flip completion
            self.ep_reward_penalty -= 20.0
            terminated = True # Dead is dead; it shouldn't roll on its head

        # 5 Boost Jump Height: Encourage the agent to jump higher by rewarding NEW peak heights
        jump_rew = 0.0
        try:
            hull_y = self.env.unwrapped.hull.position.y
            height_above_ground = max(hull_y - 1.4, 0.0)
            if height_above_ground > self.episode_max_height:
                # Reward difference between previous max and new max
                jump_rew = (height_above_ground - self.episode_max_height) * 20.0
                self.episode_max_height = height_above_ground
        except AttributeError:
            pass
            
        custom_reward += jump_rew
        self.ep_reward_jump += jump_rew

        # 5 Reward using leg contact: Encourage the agent to use its legs for powerful jumps
        leg_rew = 0.0
        if obs[8]:  # leg1 ground contact
            leg_rew = 0.1
        if obs[13]: # leg2 ground contact
            leg_rew = 0.1
            
        custom_reward += leg_rew
        self.ep_reward_legs += leg_rew

        # 6. Angular Velocity Reward: Reward for spinning fast
        # obs[1] is hull angular velocity. Because a frontflip produces a negative angular velocity
        # in the Box2D engine, we invert it so that faster forward spinning gives a positive value.
        ang_vel_rew = 0.0
        if not self.flip_completed:  # Only reward spinning *during* the flip (not after landing or on the ground)
            forward_ang_vel = -obs[1] 
            if forward_ang_vel > 0:
                ang_vel_rew = forward_ang_vel * 10.0  # Encourage carrying angular momentum
                
        custom_reward += ang_vel_rew
        self.ep_reward_ang_vel += ang_vel_rew

        # Optional: penalize large motor torques
        #custom_reward -= sum(abs(action)) * 0.01

        # On episode end, surface episode-level metrics in info for the callback to read
        if terminated or truncated:
            info["episode_metrics"] = {
                "flip_completed": float(self.flip_completed),
                "landed": float(self.landed),
                "max_rotation_rad": self.episode_max_rotation,
                "max_rotation_deg": np.degrees(self.episode_max_rotation),
                "max_rotation_pct": self.episode_max_rotation / (2 * np.pi) * 100,  # % of full flip
                "max_height": self.episode_max_height,
                "episode_length": self.step_counter,
                "reward_rotation": self.ep_reward_rotation,
                "reward_penalty": self.ep_reward_penalty,
                "reward_jump": self.ep_reward_jump,
                "reward_legs": self.ep_reward_legs,
                "reward_landing": self.ep_reward_landing,
                "reward_ang_vel": self.ep_reward_ang_vel,
            }

        return obs, custom_reward, terminated, truncated, info


if __name__ == "__main__":
    import os
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv
    from stable_baselines3.common.callbacks import BaseCallback, CallbackList

    class RenderCallback(BaseCallback):
        """Periodically pauses training to render the current policy for visual inspection."""
        def __init__(self, render_freq: int, verbose=0):
            super().__init__(verbose)
            self.render_freq = render_freq
            self.render_env = None

        def _on_step(self) -> bool:
            if self.num_timesteps > 0 and self.num_timesteps % self.render_freq == 0:
                print(f"\n--- [Step {self.num_timesteps}] Pausing to demonstrate current policy ---")
                if self.render_env is None:
                    e = gym.make("BipedalWalker-v3", hardcore=False, render_mode="human")
                    self.render_env = BipedalFlipperWrapper(e)

                obs, _ = self.render_env.reset()
                done = False
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, _, terminated, truncated, _ = self.render_env.step(action)
                    done = terminated or truncated

                print("--- Demonstration over. Resuming training ---\n")
            return True

    class FlipMetricsCallback(BaseCallback):
        """
        Reads per-episode metrics from the wrapper's info dict and logs them
        to TensorBoard under the 'flip/' prefix. This gives live graphs of:
          - flip/success_rate         : fraction of episodes that completed the flip
          - flip/max_rotation_pct     : how far toward a full flip the agent got (%)
          - flip/max_rotation_deg     : same, in degrees
          - flip/max_height           : peak height above ground
          - flip/episode_length       : steps taken per episode
        All values are averaged over the last `window` completed episodes.
        """
        def __init__(self, window: int = 100, verbose=0):
            super().__init__(verbose)
            self.window = window
            # Circular buffers — one entry per completed episode
            self._flip_success: list[float] = []
            self._landed_success: list[float] = []
            self._max_rotation_pct: list[float] = []
            self._max_rotation_deg: list[float] = []
            self._max_height: list[float] = []
            self._ep_length: list[float] = []
            self._rew_rotation: list[float] = []
            self._rew_penalty: list[float] = []
            self._rew_jump: list[float] = []
            self._rew_legs: list[float] = []
            self._rew_landing: list[float] = []
            self._rew_ang_vel: list[float] = []

        def _on_step(self) -> bool:
            # self.locals["infos"] is a list with one entry per parallel env
            for info in self.locals.get("infos", []):
                ep = info.get("episode_metrics")
                if ep is None:
                    continue
                self._flip_success.append(ep["flip_completed"])
                self._landed_success.append(ep["landed"])
                self._max_rotation_pct.append(ep["max_rotation_pct"])
                self._max_rotation_deg.append(ep["max_rotation_deg"])
                self._max_height.append(ep["max_height"])
                self._ep_length.append(ep["episode_length"])
                self._rew_rotation.append(ep["reward_rotation"])
                self._rew_penalty.append(ep["reward_penalty"])
                self._rew_jump.append(ep["reward_jump"])
                self._rew_legs.append(ep["reward_legs"])
                self._rew_landing.append(ep["reward_landing"])
                self._rew_ang_vel.append(ep["reward_ang_vel"])

                # Keep only the last `window` episodes
                for buf in (self._flip_success, self._landed_success, self._max_rotation_pct,
                            self._max_rotation_deg, self._max_height, self._ep_length,
                            self._rew_rotation, self._rew_penalty, self._rew_jump,
                            self._rew_legs, self._rew_landing, self._rew_ang_vel):
                    if len(buf) > self.window:
                        buf.pop(0)

            # Log rolling averages every step (TensorBoard will smooth them anyway)
            if self._flip_success:
                self.logger.record("flip/success_rate",     np.mean(self._flip_success))
                self.logger.record("flip/landed_rate",      np.mean(self._landed_success))
                self.logger.record("flip/max_rotation_pct", np.mean(self._max_rotation_pct))
                self.logger.record("flip/max_rotation_deg", np.mean(self._max_rotation_deg))
                self.logger.record("flip/max_height",       np.mean(self._max_height))
                self.logger.record("flip/episode_length",   np.mean(self._ep_length))
                self.logger.record("flip_rewards/rotation", np.mean(self._rew_rotation))
                self.logger.record("flip_rewards/penalty",  np.mean(self._rew_penalty))
                self.logger.record("flip_rewards/jump",     np.mean(self._rew_jump))
                self.logger.record("flip_rewards/legs",     np.mean(self._rew_legs))
                self.logger.record("flip_rewards/landing",  np.mean(self._rew_landing))
                self.logger.record("flip_rewards/ang_vel",  np.mean(self._rew_ang_vel))
            return True


    print("Setting up vectorized environments...")

    def make_env():
        def _init():
            e = gym.make("BipedalWalker-v3", hardcore=False)
            e = BipedalFlipperWrapper(e)
            return e
        return _init

    num_envs = 32
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Creating {num_envs} parallel environments...")
    vec_env = SubprocVecEnv([make_env() for _ in range(num_envs)])

    # PPO hyperparams tuned for 32 parallel envs + GPU:
    #   n_steps=2048 → rollout buffer = 32 * 2048 = 65,536 transitions per update
    #   batch_size=4096 → large GPU minibatches for high throughput
    #   ent_coef=0.01 → encourage exploration for a hard acrobatic task
    model_path = "ppo_bipedal_flipper.zip"
    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}...")
        model = PPO.load(model_path, env=vec_env, device=device,
                         tensorboard_log="./tb_logs")
    else:
        print(f"Initializing new PPO model on {device}...")
        model = PPO(
            "MlpPolicy",
            vec_env,
            verbose=1,
            device=device,
            learning_rate=3e-4,
            n_steps=2048,          # Steps per env before each update
            batch_size=4096,       # Large minibatch → saturates GPU
            n_epochs=10,           # Gradient passes over each rollout
            gae_lambda=0.95,       # GAE for variance reduction
            gamma=0.99,
            clip_range=0.2,
            ent_coef=0.1,         # Entropy bonus — critical for exploration
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=dict(
                net_arch=dict(pi=[256, 256], vf=[256, 256])  # Larger nets for GPU
            ),
            tensorboard_log="./tb_logs",
        )

    total_steps = 2_000_000   # Frontflips are hard; give it more steps
    print(f"Starting training for {total_steps:,} steps...")

    render_cb  = RenderCallback(render_freq=10000)
    metrics_cb = FlipMetricsCallback(window=200)
    callbacks  = CallbackList([render_cb, metrics_cb])

    print("To monitor training live, open a new terminal and run:")
    print("  tensorboard --logdir ./tb_logs")
    print("Then open http://localhost:6006 in your browser.\n")

    model.learn(total_timesteps=total_steps, callback=callbacks, progress_bar=True)

    print("Training finished! Saving model...")
    model.save("ppo_bipedal_flipper")
    vec_env.close()

    # To test after training, run: python test_model.py
