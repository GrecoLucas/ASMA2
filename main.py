import gymnasium as gym

def run():
    # Usando o CarRacing que é nativo e não precisa de Box2D
    env = gym.make("CarRacing-v3", render_mode="human", lap_complete_percent=0.95, domain_randomize=False, continuous=False)

    print("Action space:", env.action_space)
    print("Observation space:", env.observation_space)

    state, info = env.reset()
    terminated = False
    truncated = False
    step_count = 0

    print("Iniciando CarRacing...")

    while not (terminated or truncated):
        action = env.action_space.sample()
        state, reward, terminated, truncated, info = env.step(action)
        step_count += 1

    env.close()
    print(f"Jogo terminado em {step_count} passos.")

if __name__ == "__main__":
    run()