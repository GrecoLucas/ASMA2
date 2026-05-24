import gymnasium as gym
from stable_baselines3 import PPO
from bipedal_flipper import BipedalFlipperWrapper

def main():
    print("Loading test environment...")
    # Create the environment with rendering enabled
    test_env = gym.make("BipedalWalker-v3", hardcore=False, render_mode="human")
    test_env = BipedalFlipperWrapper(test_env)
    
    model_path = "ppo_bipedal_flipper.zip"
    try:
        print(f"Loading model from {model_path}...")
        model = PPO.load(model_path, env=test_env, device="cpu")
    except Exception as e:
        print(f"Could not load model. Make sure it exists. Error: {e}")
        return

    print("Testing trained agent. Close the window to stop.")
    obs, info = test_env.reset()
    
    while True:
        # Use deterministic=True for evaluation to get the best actions
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(action)
        
        if terminated or truncated:
            obs, info = test_env.reset()

if __name__ == "__main__":
    main()
