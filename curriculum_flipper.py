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
    Four-stage curriculum wrapper that trains a BipedalWalker to perform
    clean backflips and land feet-first.

    Stage 1  — Rotation Mastery  (gravity = -5.0)
        Goal : discover and reliably perform a full backflip.
        Logic: fall penalty is neutralised so the agent takes risks.
               Rewards raw angular speed, monotone rotation progress,
               airtime, and milestone completions (90°, 180°, 270°, 360°).

    Stage 2  — Landing           (gravity = -7.5)
        Goal : land upright on feet after the flip.
        Logic: knee-crash landings (feet touch but hull not upright) are
               penalised and terminate the episode.
               Clean landings (hull angle < 0.4 rad) give a big bonus.
               Continuous uprightness reward + tuck reward guide the agent.

    Stage 3  — Consolidation     (gravity = -10.0, real physics)
        Goal : perform the full sequence under realistic gravity.
        Logic: same as Stage 2 but with higher bonuses/penalties and
               no fall-penalty neutralisation.

    Stage 4  — Precision Landing (gravity = -10.0, tighter standards)
        Goal : land cleanly on BOTH feet with legs extended and minimal
               impact velocity, hull nearly upright.
        Logic:
          • Clean landing requires BOTH feet simultaneously (not just one).
          • Tighter hull angle threshold: 0.25 rad (≈14°) instead of 0.4.
          • Post-flip descent: rewards legs pointing DOWN (hips extended,
            low knee angle) so the agent preps a proper leg-first landing.
          • Vertical-impact penalty: landing with high downward velocity
            gives a penalty proportional to speed (hard crash = bad).
          • Hip-down posture reward: while airborne after flip, positive
            reward for hip joints angled to point legs toward the ground.
          • Larger clean-landing bonus (3000) to dominate policy updates.
          • Harsher crash penalty (-200) to strongly discourage knee-lands.
    """

    # ── Gravity per stage ──────────────────────────────────────────────────
    GRAVITY = {1: -5.0, 2: -7.5, 3: -10.0, 4: -10.0}

    # ── Landing angle thresholds (radians) ────────────────────────────────
    #   Hull angle ≈ 0  → upright
    #   Hull angle ≈ ±π → fully upside-down
    CLEAN_LANDING_ANGLE = 0.4   # ≈ 23° off vertical → success (stages 1-3)
    CLEAN_LANDING_ANGLE_S4 = 0.25  # ≈ 14° off vertical → success (stage 4)
    CRASH_LANDING_ANGLE = 0.8   # ≈ 46° off vertical → knee crash

    def __init__(self, env, stage: int = 1, max_steps: int = 1500):
        super().__init__(env)
        self.stage = stage
        self.max_steps = max_steps

        # Per-episode state (initialised in reset)
        self.cumulative_angle = 0.0
        self.prev_angle       = 0.0
        self.flip_completed   = False
        self.landed           = False
        self.step_counter     = 0
        self._milestone_flags: dict = {}

        # Expand observation: append normalised flip progress [0, ∞)
        low  = np.append(self.env.observation_space.low,  -np.inf)
        high = np.append(self.env.observation_space.high,  np.inf)
        self.observation_space = gym.spaces.Box(low, high, dtype=np.float32)

    # ── Reset ──────────────────────────────────────────────────────────────
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        # Apply stage-dependent gravity (gets closer to real gravity each stage)
        gravity = self.GRAVITY.get(self.stage, -5.0)
        try:
            self.env.unwrapped.world.gravity = (0.0, float(gravity))
        except Exception:
            pass

        # Reset all per-episode tracking
        self.cumulative_angle = 0.0
        self.prev_angle       = obs[0]
        self.flip_completed   = False
        self.landed           = False
        self.step_counter     = 0
        self._milestone_flags = {}

        obs_out = np.append(obs, 0.0).astype(np.float32)
        return obs_out, info

    # ── Step ───────────────────────────────────────────────────────────────
    def step(self, action):
        obs, base_reward, terminated, truncated, info = self.env.step(action)
        self.step_counter += 1

        # ── Angle tracking ──────────────────────────────────────────────
        current_angle = obs[0]
        delta_angle   = current_angle - self.prev_angle

        # Wrap delta to [-π, π] to handle the ±π boundary
        if delta_angle >  np.pi:
            delta_angle -= 2 * np.pi
        elif delta_angle < -np.pi:
            delta_angle += 2 * np.pi

        prev_cumulative        = self.cumulative_angle
        self.cumulative_angle += delta_angle
        self.prev_angle        = current_angle

        abs_angle = abs(self.cumulative_angle)
        abs_prev  = abs(prev_cumulative)

        # ── Useful observations ─────────────────────────────────────────
        hull_angle   = obs[0]          # ≈ 0 = upright, ±π = upside-down
        ang_vel      = obs[1]          # hull angular velocity
        foot1_down   = obs[8]  == 1.0  # leg 1 lower contact
        foot2_down   = obs[13] == 1.0  # leg 2 lower contact
        feet_contact = foot1_down or foot2_down
        both_feet    = foot1_down and foot2_down
        in_air       = not foot1_down and not foot2_down
        is_falling   = (base_reward == -100)  # hull hit the ground

        # knee joint angles (obs[6], obs[11] ≈ 0 = extended, ~2 = fully tucked)
        knee1_angle = obs[6]
        knee2_angle = obs[11]

        custom_reward = 0.0

        # ════════════════════════════════════════════════════════════════
        # PRE-FLIP rewards (all stages, only until flip is complete)
        # ════════════════════════════════════════════════════════════════
        if not self.flip_completed:
            # 1. Raw angular speed — spin faster!
            custom_reward += abs(ang_vel) * 5.0

            # 2. Monotone rotation progress
            #    Using abs() means wiggling back-and-forth yields NO progress
            rotation_progress = abs_angle - abs_prev
            custom_reward += rotation_progress * 15.0

            # 3. Airtime bonus during flip
            if in_air:
                custom_reward += 1.0

            # 4. Milestone bonuses: 25%, 50%, 75% of full rotation
            milestones = [np.pi / 2, np.pi, 3 * np.pi / 2]
            for ms in milestones:
                key = f"ms_{ms:.4f}"
                if not self._milestone_flags.get(key, False) and abs_angle >= ms:
                    self._milestone_flags[key] = True
                    custom_reward += 50.0

        # ════════════════════════════════════════════════════════════════
        # FLIP COMPLETION (2π = one full rotation)
        # ════════════════════════════════════════════════════════════════
        if abs_angle >= 2 * np.pi and not self.flip_completed:
            self.flip_completed = True
            custom_reward += 300.0

        # ════════════════════════════════════════════════════════════════
        # POST-FLIP rewards (all stages, once flip_completed)
        # ════════════════════════════════════════════════════════════════
        if self.flip_completed and not self.landed:

            # 1. Strong continuous uprightness reward
            #    cos(0) = 1.0 (upright, max reward)
            #    cos(π) = -1.0 (upside-down, max penalty)
            #    This creates a smooth gradient toward standing
            upright_rew = np.cos(hull_angle) * 25.0
            custom_reward += upright_rew

            # 2. Dampen residual spin — we want the agent to STOP rotating and land
            custom_reward -= abs(ang_vel) * 5.0

            # 3. In-air tuck reward: bent knees give control during descent
            #    (Stage 4 replaces this with leg-extension reward below)
            if in_air and self.stage < 4:
                tuck = knee1_angle + knee2_angle  # higher = more bent
                custom_reward += tuck * 1.5

            # ── Stage 4 post-flip in-air shaping ───────────────────────
            if self.stage == 4 and in_air:
                # a) Hip-down posture: reward hips angled to point legs toward ground.
                #    obs[4] = joint[0].angle (hip 1), obs[9] = joint[2].angle (hip 2)
                #    Positive hip angle means leg sweeps forward/down in landing pose.
                #    We want both hips extended downward → angle close to -0.5..0 rad.
                hip1_angle = obs[4]
                hip2_angle = obs[9]
                # Reward when both hips near 0 (legs hanging straight down)
                hip_posture = 2.0 - abs(hip1_angle) - abs(hip2_angle)  # max=2 when both at 0
                custom_reward += hip_posture * 3.0

                # b) Leg extension reward: discourage tucked knees during descent.
                #    knee_angle ≈ 0 → extended, ≈ 2 → fully tucked.
                #    Reward for LOW knee angle (legs straight, ready to absorb impact).
                extension = 4.0 - (knee1_angle + knee2_angle)  # max=4 when fully extended
                custom_reward += extension * 2.0

            # 4. KNEE-CRASH PENALTY: feet touch but hull is NOT upright
            clean_angle = self.CLEAN_LANDING_ANGLE_S4 if self.stage == 4 \
                          else self.CLEAN_LANDING_ANGLE

            if feet_contact and abs(hull_angle) > self.CRASH_LANDING_ANGLE:
                # Landing on knees / still upside-down = bad
                crash_penalty = -100.0 if self.stage <= 2 else (-150.0 if self.stage == 3 else -200.0)
                custom_reward += crash_penalty
                terminated = True   # treat as failure, reset and retry

            # 5. CLEAN LANDING BONUS: feet touch AND hull is upright
            elif feet_contact and abs(hull_angle) < clean_angle:
                # Stage 4: require BOTH feet simultaneously for maximum precision
                landing_feet_ok = both_feet if self.stage == 4 else feet_contact

                if landing_feet_ok:
                    # Scale the bonus with how upright the hull is
                    uprightness = 1.0 - (abs(hull_angle) / clean_angle)

                    if self.stage <= 2:
                        landing_bonus = 1000.0
                    elif self.stage == 3:
                        landing_bonus = 2000.0
                    else:
                        # Stage 4: also penalise hard landings (high downward velocity)
                        vel_y = obs[3]  # normalised vertical velocity (negative = falling)
                        impact_penalty = max(0.0, -vel_y) * 50.0  # faster fall = more penalty
                        custom_reward -= impact_penalty
                        landing_bonus = 3000.0

                    custom_reward += landing_bonus * uprightness
                    self.landed = True
                    terminated  = True   # success — end the episode

                elif self.stage == 4 and not both_feet:
                    # One foot only in stage 4: small penalty, keep trying
                    custom_reward -= 20.0

        # ════════════════════════════════════════════════════════════════
        # STAGE-SPECIFIC extras
        # ════════════════════════════════════════════════════════════════
        if self.stage == 1:
            # Nullify the hard -100 fall penalty entirely.
            # The agent needs to take risks to discover the flip at all.
            if is_falling:
                custom_reward += 100.0

        elif self.stage == 2:
            # Only soften the fall penalty BEFORE the flip is done.
            # After the flip it should care about landing, not falling.
            if is_falling and not self.flip_completed:
                custom_reward += 50.0

        elif self.stage == 3:
            # Real gravity, full physics — no hand-holding.
            # The agent should already know how to flip; now it must land cleanly.
            pass

        elif self.stage == 4:
            # Same as stage 3 — no hand-holding — precision is enforced via
            # the tighter landing checks and in-air shaping above.
            pass

        # ── Step limit ──────────────────────────────────────────────────
        if self.step_counter >= self.max_steps:
            truncated = True

        # ── Info ────────────────────────────────────────────────────────
        info["flip_completed"]   = self.flip_completed
        info["landed"]           = self.landed
        info["cumulative_angle"] = self.cumulative_angle
        info["abs_angle_deg"]    = np.degrees(abs_angle)

        total_reward = base_reward + custom_reward
        obs_out = np.append(obs, abs_angle / (2 * np.pi)).astype(np.float32)
        return obs_out, total_reward, terminated, truncated, info


# ─────────────────────────────────────────────────────────────────────────────
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
    MODELS_DIR      = "models"
    os.makedirs(MODELS_DIR, exist_ok=True)
    STAGE1_SAVE     = os.path.join(MODELS_DIR, "ppo_flipper_stage1")
    STAGE2_SAVE     = os.path.join(MODELS_DIR, "ppo_flipper_stage2")
    STAGE3_SAVE     = os.path.join(MODELS_DIR, "ppo_flipper_stage3")
    STAGE4_SAVE     = os.path.join(MODELS_DIR, "ppo_flipper_stage4")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ── Callbacks ────────────────────────────────────────────────────────────

    class FlipMetricsCallback(BaseCallback):
        """Logs flip/landing stats to TensorBoard."""
        def __init__(self, window=200, verbose=0):
            super().__init__(verbose)
            self.window = window
            self._bufs: dict[str, list] = {k: [] for k in [
                "flip", "landed", "rot_deg", "ep_len"]}

        def _on_step(self) -> bool:
            for info in self.locals.get("infos", []):
                if info.get("flip_completed") is None:
                    continue
                self._bufs["flip"].append(float(info["flip_completed"]))
                self._bufs["landed"].append(float(info.get("landed", False)))
                self._bufs["rot_deg"].append(info.get("abs_angle_deg", 0.0))
                for buf in self._bufs.values():
                    if len(buf) > self.window:
                        buf.pop(0)
            if self._bufs["flip"]:
                self.logger.record("flip/success_rate",  float(np.mean(self._bufs["flip"])))
                self.logger.record("flip/landing_rate",  float(np.mean(self._bufs["landed"])))
                self.logger.record("flip/max_rot_deg",   float(np.mean(self._bufs["rot_deg"])))
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
        # Very low LR: the flip is already good, we're fine-tuning landing precision
        model.learning_rate = 3e-5
        model.ent_coef      = 0.0005  # Minimal entropy: exploit precision
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
    # TEST: run 5 episodes with the final model rendered
    # ────────────────────────────────────────────────────────────────────────
    import time
    if os.path.exists(f"{STAGE4_SAVE}.zip"):
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
