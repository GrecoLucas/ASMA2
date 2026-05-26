"""
curriculum_flipper.py
=====================
Improved 4-stage curriculum wrapper for BipedalWalker backflip training.

KEY IMPROVEMENTS over the notebook version:
  1. Gravity curriculum  — Stage 1: -5.0 (easy), Stage 2: -7.5, Stage 3/4: -10.0 (real)
  2. Fixed landing check — now requires the hull to be upright, not just feet touching
  3. Knee-crash penalty  — landing with hull angle > 0.8 rad is penalised and ends episode
  4. Strong uprightness reward after flip — cos(hull_angle)*25 gives smooth gradient toward standing
  5. In-air tuck reward  — rewards bent knees while airborne post-flip (helps rotation control)
  6. Dampen residual spin post-flip — discourages continuing to spin after the flip is done
  7. Stage 3 added       — trains under real gravity with no assistance
  8. Stage 4 added       — precision leg-first landing: both feet required, legs must extend
                           downward during descent, hard crash penalty by vertical velocity,
                           tighter clean-landing angle (0.25 rad), hip-down posture reward

HOW TO USE IN THE NOTEBOOK:
  Replace the inline CurriculumFlipperWrapper class definition with:
      from curriculum_flipper import CurriculumFlipperWrapper
  Then use exactly as before (make_env_stage1, make_env_stage2, etc.)

TO RUN STANDALONE:
  python curriculum_flipper.py
"""

import os
import gymnasium as gym
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Wrapper
# ─────────────────────────────────────────────────────────────────────────────

class CurriculumFlipperWrapper(gym.Wrapper):
    """
    5-stage curriculum wrapper for BipedalWalker backflip training.

    Stage 1 — Rotation Mastery  (gravity = -5.0)
        Learn to discover and reliably complete a full backflip.
        Fall penalty cancelled so the agent takes risks.

    Stage 2 — Landing  (gravity = -7.5)
        Land upright on feet after the flip.
        Knee-crash landings penalised and terminated.

    Stage 3 — Consolidation  (gravity = -10.0)
        Same as Stage 2 under full gravity, no hand-holding.

    Stage 4 — Precision Landing  (gravity = -10.0, strict)
        Fixes the Stage-3 problem where the flip completes too close
        to the ground so the legs end up pointing forward.
        Enforces a tighter landing angle (0.22 rad) and stability window.

    Stage 5 — Landing Stabilization  (gravity = -10.0, recovery + straight legs)
        Same as Stage 4 but removes premature knee-crash terminations to allow
        recovery training. Also requires the legs to be straight (extended) at the
        end of the stability window before the landing is successful.
    """

    GRAVITY                = {1: -5.0, 2: -7.5, 3: -10.0, 4: -10.0, 5: -10.0}
    CLEAN_LANDING_ANGLE    = 0.4    # ~23°  stages 1-3
    CLEAN_LANDING_ANGLE_S4 = 0.22   # ~12.6° stage 4
    CLEAN_LANDING_ANGLE_S5 = 0.28   # ~16°  stage 5
    CRASH_LANDING_ANGLE    = 0.8    # ~46°  knee crash (stages 1-4)
    CRASH_LANDING_ANGLE_S5 = 1.1    # ~63°  knee crash (stage 5)
    STABILITY_STEPS        = 45     # steps to hold landing pose (stages 4-5)

    def __init__(self, env, stage: int = 1, max_steps: int = 1500):
        super().__init__(env)
        self.stage     = stage
        self.max_steps = max_steps
        self.cumulative_angle = 0.0
        self.prev_angle       = 0.0
        self.flip_completed   = False
        self.landed           = False
        self.step_counter     = 0
        self._milestone_flags = {}
        self.stable_steps     = 0
        self.max_stable_steps = 0
        low  = np.append(self.env.observation_space.low,  -np.inf)
        high = np.append(self.env.observation_space.high,  np.inf)
        self.observation_space = gym.spaces.Box(low, high, dtype=np.float32)

    # ------------------------------------------------------------------
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        gravity = self.GRAVITY.get(self.stage, -5.0)
        try:
            self.env.unwrapped.world.gravity = (0.0, float(gravity))
        except Exception:
            pass
        self.cumulative_angle = 0.0
        self.prev_angle       = obs[0]
        self.flip_completed   = False
        self.landed           = False
        self.step_counter     = 0
        self._milestone_flags = {}
        self.stable_steps     = 0
        self.max_stable_steps = 0
        return np.append(obs, 0.0).astype(np.float32), info

    # ------------------------------------------------------------------
    def step(self, action):
        obs, base_reward, terminated, truncated, info = self.env.step(action)
        self.step_counter += 1

        # ── Angle tracking ──────────────────────────────────────────────
        current_angle = obs[0]
        delta_angle   = current_angle - self.prev_angle
        if delta_angle >  np.pi: delta_angle -= 2 * np.pi
        if delta_angle < -np.pi: delta_angle += 2 * np.pi
        prev_cumulative        = self.cumulative_angle
        self.cumulative_angle += delta_angle
        self.prev_angle        = current_angle
        abs_angle = abs(self.cumulative_angle)
        abs_prev  = abs(prev_cumulative)

        # ── Observation shorthands ──────────────────────────────────────
        hull_angle  = obs[0]            # 0=upright, ±π=upside-down
        ang_vel     = obs[1]            # hull angular velocity
        vel_x       = obs[2]            # normalised horizontal velocity
        vel_y       = obs[3]            # normalised vertical velocity (pos=up)
        foot1_down  = obs[8]  == 1.0
        foot2_down  = obs[13] == 1.0
        feet_contact = foot1_down or foot2_down
        both_feet    = foot1_down and foot2_down
        in_air       = not foot1_down and not foot2_down
        is_falling   = (base_reward == -100)
        knee1_angle  = obs[6]           # 0=extended, ~2=tucked
        knee2_angle  = obs[11]
        hip1_angle   = obs[4]           # leg 1 hip angle
        hip2_angle   = obs[9]           # leg 2 hip angle
        height_frac  = obs[14]          # ray-0 lidar: 0=touching ground, 1=~5.3m up

        custom_reward = 0.0

        # ══════════════════════════════════════════════════════════════
        # PRE-FLIP rewards
        # ══════════════════════════════════════════════════════════════
        if not self.flip_completed:
            custom_reward += abs(ang_vel) * 5.0               # spin faster
            custom_reward += (abs_angle - abs_prev) * 15.0   # rotation progress
            if in_air:
                custom_reward += 1.0                          # airtime bonus
            # HEIGHT FIX 1: reward upward velocity (stages 3+)
            # Encourages jumping BEFORE spinning to gain altitude first.
            if self.stage >= 3 and vel_y > 0:
                custom_reward += vel_y * 10.0
            for ms in [np.pi / 2, np.pi, 3 * np.pi / 2]:
                key = f"ms_{ms:.4f}"
                if not self._milestone_flags.get(key, False) and abs_angle >= ms:
                    self._milestone_flags[key] = True
                    custom_reward += 50.0

        # ══════════════════════════════════════════════════════════════
        # FLIP COMPLETION (full 2π rotation)
        # ══════════════════════════════════════════════════════════════
        if abs_angle >= 2 * np.pi and not self.flip_completed:
            self.flip_completed = True
            custom_reward += 300.0
            # HEIGHT FIX 2: large altitude bonus at flip completion (stages 3+)
            # obs[14]=straight-down lidar fraction: 0=ground, 1=~5.3m above.
            # Higher flip completion → more room for legs to extend downward.
            if self.stage >= 3:
                custom_reward += height_frac * 200.0

        # ══════════════════════════════════════════════════════════════
        # POST-FLIP rewards (flip done, waiting to land)
        # ══════════════════════════════════════════════════════════════
        if self.flip_completed and not self.landed:

            # 1. Uprightness gradient
            custom_reward += np.cos(hull_angle) * 25.0

            # 2. Dampen residual spin
            custom_reward -= abs(ang_vel) * 5.0

            # 2b. Anti-bounce: penalise vertical velocity unconditionally post-flip
            #     This fires whether or not feet are touching, so hopping is always costly.
            custom_reward -= abs(vel_y) * 6.0

            if in_air:
                # 3. In-air tuck (stages 1-2 only)
                if self.stage < 3:
                    custom_reward += (knee1_angle + knee2_angle) * 1.5

                # HEIGHT FIX 3: per-step altitude reward while airborne (stages 3+)
                # Keeps the agent high so it has TIME to orient legs before impact.
                if self.stage >= 3:
                    custom_reward += height_frac * 4.0

                # Stage 4/5: in-air landing prep
                if self.stage in [4, 5]:
                    if self.stable_steps > 0:
                        custom_reward -= 60.0       # penalty for breaking stability (doubled)
                        self.stable_steps = 0       # reset if went airborne mid-window
                    # a) Hip-down: legs hanging toward ground
                    hip_posture = 2.0 - abs(obs[4]) - abs(obs[9])
                    custom_reward += hip_posture * 3.0
                    # b) Leg extension: straight legs absorb impact better
                    custom_reward += (4.0 - knee1_angle - knee2_angle) * 2.0

            # ── Landing angle threshold ──────────────────────────────
            clean_angle = self.CLEAN_LANDING_ANGLE_S5 if self.stage == 5 \
                          else (self.CLEAN_LANDING_ANGLE_S4 if self.stage == 4 else self.CLEAN_LANDING_ANGLE)
            crash_angle = self.CRASH_LANDING_ANGLE_S5 if self.stage == 5 else self.CRASH_LANDING_ANGLE

            # 4. KNEE-CRASH CHECK
            if feet_contact and abs(hull_angle) > crash_angle:
                if self.stage <= 4:
                    penalty = -100.0 if self.stage <= 2 else (-150.0 if self.stage == 3 else -200.0)
                    custom_reward    += penalty
                    self.stable_steps = 0
                    terminated        = True
                else: # Stage 5: do NOT terminate! Let it try to recover.
                    custom_reward    -= 10.0  # penalty for bad posture
                    self.stable_steps = 0

            # 5. CLEAN LANDING / STABILITY WINDOW
            elif feet_contact and abs(hull_angle) < clean_angle:
                if self.stage <= 2:
                    custom_reward += 1000.0 * (1.0 - (abs(hull_angle) / clean_angle))
                    self.landed = True;  terminated = True

                elif self.stage == 3:
                    custom_reward += 2000.0 * (1.0 - (abs(hull_angle) / clean_angle))
                    self.landed = True;  terminated = True

                else: # self.stage in [4, 5]
                    # Increment stability counter while at least 1 foot has contact and hull is upright
                    self.stable_steps += 1
                    uprightness = 1.0 - (abs(hull_angle) / clean_angle)
                    custom_reward += uprightness * 20.0   # strong reward for standing upright
                    custom_reward -= abs(ang_vel) * 10.0  # no spinning
                    custom_reward -= abs(vel_x)   * 8.0   # no sliding
                    custom_reward -= abs(vel_y)   * 15.0  # no bouncing / hopping (increased)

                    # Hopping detection: feet down but still bouncing hard → reset counter
                    if abs(vel_y) > 0.35 and self.stable_steps > 0:
                        custom_reward -= 50.0
                        self.stable_steps = 0

                    # Stage 5: reward straight legs and leg spread (triangle/split stance)
                    if self.stage == 5:
                        legs_straightness = 4.0 - (knee1_angle + knee2_angle)
                        custom_reward += legs_straightness * 5.0  # strong incentive to extend legs
                        
                        # Encourage split stance (legs spread out like a triangle to increase stability)
                        # Hip spread: sum of absolute angles rewards both legs angling outward.
                        # Using the difference was unreliable when both hips are at similar non-zero angles.
                        hip_spread = abs(hip1_angle) + abs(hip2_angle)
                        custom_reward += min(hip_spread, 1.2) * 8.0   # reward up to +9.6 for wide stance

                    # Mid-window milestone at half-way to give a shaping signal
                    half_window = self.STABILITY_STEPS // 2
                    if self.stable_steps == half_window:
                        custom_reward += 500.0 * uprightness  # big intermediate reward

                    # COMPLETION: stable for full window with feet on the ground
                    # Stage 4: requires both feet; Stage 5: only 1 foot needed (more forgiving)
                    landing_feet = both_feet if self.stage == 4 else feet_contact
                    if self.stable_steps >= self.STABILITY_STEPS and landing_feet:
                        # Straight-leg bonus for Stage 5 (reward-only, not gating)
                        knee_bonus = 0.0
                        if self.stage == 5:
                            avg_knee = (knee1_angle + knee2_angle) / 2.0
                            knee_bonus = max(0.0, (0.7 - avg_knee)) * 500.0  # up to +350 for straight legs
                        custom_reward -= max(0.0, -vel_y) * 50.0  # impact penalty
                        custom_reward += 3000.0 * uprightness + knee_bonus
                        self.landed   = True
                        terminated    = True

            # 6. BREAKING STABILITY / RESET (Stage 4/5 only)
            elif self.stage in [4, 5]:
                # Hull angle not clean and not in crash zone: penalise and reset counter
                if self.stable_steps > 0:
                    custom_reward -= 20.0  # softer penalty (was 30) so partial progress still counts
                    self.stable_steps = 0

        # Update peak stable steps
        if self.stable_steps > self.max_stable_steps:
            self.max_stable_steps = self.stable_steps

        # ══════════════════════════════════════════════════════════════
        # STAGE-SPECIFIC EXTRAS
        # ══════════════════════════════════════════════════════════════
        if self.stage == 1:
            if is_falling:
                custom_reward += 100.0
        elif self.stage == 2:
            if is_falling and not self.flip_completed:
                custom_reward += 50.0
        elif self.stage >= 3:
            pass  # real gravity, no hand-holding

        if self.step_counter >= self.max_steps:
            truncated = True

        info["flip_completed"]   = self.flip_completed
        info["landed"]           = self.landed
        info["cumulative_angle"] = self.cumulative_angle
        info["abs_angle_deg"]    = np.degrees(abs_angle)
        info["max_stable_steps"] = self.max_stable_steps
        info["knee1_angle"]      = knee1_angle
        info["knee2_angle"]      = knee2_angle

        obs_out = np.append(obs, abs_angle / (2 * np.pi)).astype(np.float32)
        return obs_out, base_reward + custom_reward, terminated, truncated, info


# Standalone training script
# Run with: python curriculum_flipper.py
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv
    from stable_baselines3.common.callbacks import BaseCallback, CallbackList
    import custom_bipedal

    # ── Config ──────────────────────────────────────────────────────────────
    NUM_ENVS        = 32
    STAGE1_STEPS    = 5_000_000
    STAGE2_STEPS    = 4_000_000
    STAGE3_STEPS    = 3_000_000
    STAGE4_STEPS    = 3_000_000
    STAGE5_STEPS    = 1_500_000
    MODELS_DIR      = "models"
    os.makedirs(MODELS_DIR, exist_ok=True)
    STAGE1_SAVE     = os.path.join(MODELS_DIR, "ppo_flipper_stage1")
    STAGE2_SAVE     = os.path.join(MODELS_DIR, "ppo_flipper_stage2")
    STAGE3_SAVE     = os.path.join(MODELS_DIR, "ppo_flipper_stage3")
    STAGE4_SAVE     = os.path.join(MODELS_DIR, "ppo_flipper_stage4")
    STAGE5_SAVE     = os.path.join(MODELS_DIR, "ppo_flipper_stage5")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ── Callbacks ────────────────────────────────────────────────────────────

    class FlipMetricsCallback(BaseCallback):
        """Logs flip/landing success rates and Stage 5 metrics to TensorBoard per episode."""
        def __init__(self, window=200, verbose=0):
            super().__init__(verbose)
            self.window = window
            self._flip:        list = []
            self._landed:      list = []
            self._rot_deg:     list = []
            self._stable:      list = []
            self._knee_bend:   list = []

        def _on_step(self) -> bool:
            dones = self.locals.get("dones")
            for i, info in enumerate(self.locals.get("infos", [])):
                if dones is not None and dones[i]:
                    self._flip.append(float(info.get("flip_completed", False)))
                    self._landed.append(float(info.get("landed", False)))
                    self._rot_deg.append(info.get("abs_angle_deg", 0.0))
                    self._stable.append(float(info.get("max_stable_steps", 0)))
                    
                    k1 = info.get("knee1_angle", 0.0)
                    k2 = info.get("knee2_angle", 0.0)
                    self._knee_bend.append(float(k1 + k2) / 2.0)

                    for buf in (self._flip, self._landed, self._rot_deg, self._stable, self._knee_bend):
                        if len(buf) > self.window:
                            buf.pop(0)

            if self._flip:
                self.logger.record("flip/success_rate",        np.mean(self._flip))
                self.logger.record("flip/landing_rate",        np.mean(self._landed))
                self.logger.record("flip/avg_rotation_deg",    np.mean(self._rot_deg))
                self.logger.record("flip/avg_max_stable_steps", np.mean(self._stable))
                self.logger.record("flip/avg_knee_bend",       np.mean(self._knee_bend))
            return True

    class RenderCallback(BaseCallback):
        """Periodically renders the current policy for visual inspection."""
        def __init__(self, render_freq=100_000, stage=1, verbose=0):
            super().__init__(verbose)
            self.render_freq = render_freq
            self.stage = stage
            self._render_env = None

        def _on_step(self) -> bool:
            if self.num_timesteps > 0 and self.num_timesteps % self.render_freq == 0:
                print(f"\n[Step {self.num_timesteps}] Rendering policy...")
                if self._render_env is None:
                    e = custom_bipedal.BipedalWalker(render_mode="human")
                    self._render_env = CurriculumFlipperWrapper(e, stage=self.stage)
                obs, _ = self._render_env.reset()
                done = False
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, _, terminated, truncated, _ = self._render_env.step(action)
                    done = terminated or truncated
                print("Render done. Resuming training.\n")
            return True

    # ── Environment factory ─────────────────────────────────────────────────

    def make_env(stage):
        def _init():
            e = custom_bipedal.BipedalWalker(hardcore=False)
            return CurriculumFlipperWrapper(e, stage=stage)
        return _init

    # ────────────────────────────────────────────────────────────────────────
    # STAGE 1 — Rotation Mastery  (gravity = -5.0)
    # ────────────────────────────────────────────────────────────────────────
    stage1_exists = os.path.exists(f"{STAGE1_SAVE}.zip")

    if not stage1_exists:
        print("\n" + "="*60)
        print("STAGE 1: Rotation Mastery  (gravity = -5.0)")
        print("="*60)
        vec_env_s1 = SubprocVecEnv([make_env(stage=1) for _ in range(NUM_ENVS)])
        model = PPO(
            "MlpPolicy", vec_env_s1,
            verbose=1, device=device,
            n_steps=2048,
            batch_size=8192,
            n_epochs=5,
            learning_rate=3e-4,
            gae_lambda=0.95,
            gamma=0.99,
            clip_range=0.2,
            ent_coef=0.01,    # Some entropy for exploration
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
            tensorboard_log="./tb_logs_flipper",
        )
        cbs = CallbackList([
            FlipMetricsCallback(window=200),
            RenderCallback(render_freq=100_000, stage=1),
        ])
        model.learn(total_timesteps=STAGE1_STEPS, callback=cbs, progress_bar=True)
        model.save(STAGE1_SAVE)
        vec_env_s1.close()
        print(f"Stage 1 complete. Saved to {STAGE1_SAVE}.zip")
    else:
        print(f"Stage 1 model found ({STAGE1_SAVE}.zip) — skipping training.")

    # ────────────────────────────────────────────────────────────────────────
    # STAGE 2 — Landing  (gravity = -7.5)
    # ────────────────────────────────────────────────────────────────────────
    stage2_exists = os.path.exists(f"{STAGE2_SAVE}.zip")

    if not stage2_exists:
        print("\n" + "="*60)
        print("STAGE 2: Landing  (gravity = -7.5)")
        print("="*60)
        vec_env_s2 = SubprocVecEnv([make_env(stage=2) for _ in range(NUM_ENVS)])
        model = PPO.load(f"{STAGE1_SAVE}", env=vec_env_s2, device=device,
                         tensorboard_log="./tb_logs_flipper")
        # Lower LR for fine-tuning
        model.learning_rate = 1e-4
        model.ent_coef      = 0.005  # Less entropy: exploit learned rotation
        cbs = CallbackList([
            FlipMetricsCallback(window=200),
            RenderCallback(render_freq=100_000, stage=2),
        ])
        model.learn(total_timesteps=STAGE2_STEPS, callback=cbs, progress_bar=True,
                    reset_num_timesteps=False)
        model.save(STAGE2_SAVE)
        vec_env_s2.close()
        print(f"Stage 2 complete. Saved to {STAGE2_SAVE}.zip")
    else:
        print(f"Stage 2 model found ({STAGE2_SAVE}.zip) — skipping training.")

    # ────────────────────────────────────────────────────────────────────────
    # STAGE 3 — Consolidation under real gravity  (gravity = -10.0)
    # ────────────────────────────────────────────────────────────────────────
    stage3_exists = os.path.exists(f"{STAGE3_SAVE}.zip")

    if not stage3_exists:
        print("\n" + "="*60)
        print("STAGE 3: Real gravity consolidation  (gravity = -10.0)")
        print("="*60)
        vec_env_s3 = SubprocVecEnv([make_env(stage=3) for _ in range(NUM_ENVS)])
        model = PPO.load(f"{STAGE2_SAVE}", env=vec_env_s3, device=device,
                         tensorboard_log="./tb_logs_flipper")
        model.learning_rate = 5e-5
        model.ent_coef      = 0.001
        cbs = CallbackList([
            FlipMetricsCallback(window=200),
            RenderCallback(render_freq=100_000, stage=3),
        ])
        model.learn(total_timesteps=STAGE3_STEPS, callback=cbs, progress_bar=True,
                    reset_num_timesteps=False)
        model.save(STAGE3_SAVE)
        vec_env_s3.close()
        print(f"Stage 3 complete. Saved to {STAGE3_SAVE}.zip")
    else:
        print(f"Stage 3 model found ({STAGE3_SAVE}.zip) — skipping training.")

    # ────────────────────────────────────────────────────────────────────────
    # STAGE 4 — Precision Landing  (gravity = -10.0, tighter standards)
    # ────────────────────────────────────────────────────────────────────────
    stage4_exists = os.path.exists(f"{STAGE4_SAVE}.zip")

    if not stage4_exists:
        print("\n" + "="*60)
        print("STAGE 4: Precision Landing  (gravity = -10.0, strict)")
        print("="*60)
        vec_env_s4 = SubprocVecEnv([make_env(stage=4) for _ in range(NUM_ENVS)])
        model = PPO.load(f"{STAGE3_SAVE}", env=vec_env_s4, device=device,
                         tensorboard_log="./tb_logs_flipper")
        # Slightly higher entropy to escape the reward-hacking local optimum
        model.learning_rate = 5e-5
        model.ent_coef      = 0.002   # Enough exploration to find the stability window
        cbs = CallbackList([
            FlipMetricsCallback(window=200),
            RenderCallback(render_freq=100_000, stage=4),
        ])
        model.learn(total_timesteps=STAGE4_STEPS, callback=cbs, progress_bar=True,
                    reset_num_timesteps=False)
        model.save(STAGE4_SAVE)
        vec_env_s4.close()
        print(f"Stage 4 complete. Saved to {STAGE4_SAVE}.zip")
    else:
        print(f"Stage 4 model found ({STAGE4_SAVE}.zip) — skipping training.")

    # ────────────────────────────────────────────────────────────────────────
    # STAGE 5 — Landing Stabilization  (gravity = -10.0, relaxed angles)
    # ────────────────────────────────────────────────────────────────────────
    stage5_exists = os.path.exists(f"{STAGE5_SAVE}.zip")

    if not stage5_exists:
        print("\n" + "="*60)
        print("STAGE 5: Landing Stabilization  (gravity = -10.0, relaxed)")
        print("="*60)
        vec_env_s5 = SubprocVecEnv([make_env(stage=5) for _ in range(NUM_ENVS)])
        model = PPO.load(f"{STAGE4_SAVE}", env=vec_env_s5, device=device,
                         tensorboard_log="./tb_logs_flipper")
        model.learning_rate = 5e-5
        model.ent_coef      = 0.001
        cbs = CallbackList([
            FlipMetricsCallback(window=200),
            RenderCallback(render_freq=100_000, stage=5),
        ])
        model.learn(total_timesteps=STAGE5_STEPS, callback=cbs, progress_bar=True,
                    reset_num_timesteps=False)
        model.save(STAGE5_SAVE)
        vec_env_s5.close()
        print(f"Stage 5 complete. Saved to {STAGE5_SAVE}.zip")
    else:
        print(f"Stage 5 model found ({STAGE5_SAVE}.zip) — skipping training.")

    # ────────────────────────────────────────────────────────────────────────
    # TEST: run 5 episodes with the final model rendered
    # ────────────────────────────────────────────────────────────────────────
    import time
    if os.path.exists(f"{STAGE5_SAVE}.zip"):
        final_model_path = f"{STAGE5_SAVE}.zip"
        stage = 5
    elif os.path.exists(f"{STAGE4_SAVE}.zip"):
        final_model_path = f"{STAGE4_SAVE}.zip"
        stage = 4
    elif os.path.exists(f"{STAGE3_SAVE}.zip"):
        final_model_path = f"{STAGE3_SAVE}.zip"
        stage = 3
    else:
        final_model_path = f"{STAGE2_SAVE}.zip"
        stage = 2
    print(f"\nTesting model: {final_model_path}")
    model = PPO.load(final_model_path, device=device)
    env_test = custom_bipedal.BipedalWalker(hardcore=False, render_mode="human")
    env_test = CurriculumFlipperWrapper(env_test, stage=stage)

    for ep in range(1, 6):
        obs, _ = env_test.reset()
        done = False; total_r = 0.0; steps = 0
        print(f"\n=== Test Episode {ep} ===")
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, terminated, truncated, info = env_test.step(action)
            done = terminated or truncated
            total_r += r; steps += 1
        print(f"  Steps: {steps}  |  Reward: {total_r:.1f}  |  "
              f"Flip: {info['flip_completed']}  |  Landed: {info['landed']}  |  "
              f"Rotation: {info['abs_angle_deg']:.1f}°")
        time.sleep(0.3)

    env_test.close()
    print("\nDone!")
