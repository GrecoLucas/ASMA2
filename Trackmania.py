import os
import ctypes
import argparse
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import tmrl
from stable_baselines3 import SAC

# ==========================================
# CONFIGURAÇÕES
# ==========================================
TOTAL_PASSOS = 50_000
PASTA_MODELOS = "./modelos"
MODELO_FINAL = os.path.join(PASTA_MODELOS, "trackmania_visao_sac")

LARGURA_RESTAURADA = 1920
ALTURA_RESTAURADA = 1080

# ==========================================
# WRAPPER DO AMBIENTE (A MÁGICA DA VISÃO)
# ==========================================

class TmrlVisaoWrapper(gym.Wrapper):
    """
    Separa a imagem e a telemetria num Dicionário para a Rede Convolucional.
    """
    def __init__(self, env):
        super().__init__(env)
        
        # Aqui dizemos à IA: "Você vai receber duas coisas diferentes"
        self.observation_space = spaces.Dict({
            # A Câmera: 4 imagens empilhadas de 64x64 pixels
            "camera": spaces.Box(low=0.0, high=255.0, shape=(4, 64, 64), dtype=np.float32),
            # Sensores: 9 números de telemetria
            "sensores": spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32)
        })

    def _processar_obs(self, obs):
        # 1. Extrai a câmera do jogo (índice 3 do TMRL)
        camera = np.array(obs[3], dtype=np.float32)
        
        # 2. Extrai e normaliza os sensores
        try:
            speed = np.array(obs[0], dtype=np.float32).flatten() / 300.0
            gear  = np.array(obs[1], dtype=np.float32).flatten() / 6.0
            rpm   = np.tanh(np.array(obs[2], dtype=np.float32).flatten() / 10000.0)
            vel1  = np.array(obs[4], dtype=np.float32).flatten()
            vel2  = np.array(obs[5], dtype=np.float32).flatten()
            sensores = np.concatenate([speed, gear, rpm, vel1, vel2]).astype(np.float32)
        except Exception:
            sensores = np.zeros(9, dtype=np.float32)
            
        return {"camera": camera, "sensores": sensores}

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._processar_obs(obs), info

    def step(self, action):
        obs, recompensa_base, terminated, truncated, info = self.env.step(action)
        
        # A base do TMRL já mede o avanço real na pista. Vamos usar isso!
        recompensa_final = recompensa_base
        
        velocidade_bruta = np.array(obs[0], dtype=np.float32).item()
        
        # Punição severa: Se estiver muito devagar ou a fazer piões sem progredir
        if recompensa_base <= 0.01 and not (terminated or truncated):
            recompensa_final -= 1.0 # Dói muito mais ficar a rodar em círculos agora
            
        # Punições finais
        if terminated or truncated:
            if recompensa_base > 0.5: # Significa que avançou bem antes de acabar
                recompensa_final += 20.0
            else: # Bateu logo no início ou não saiu do sítio
                recompensa_final -= 20.0
                
        return self._processar_obs(obs), recompensa_final, terminated, truncated, info


# ==========================================
# FUNÇÕES PRINCIPAIS
# ==========================================

def restaurar_janela():
    try:
        user32 = ctypes.windll.user32
        hwnd = user32.FindWindowW(None, "Trackmania")
        if hwnd:
            user32.MoveWindow(hwnd, 0, 0, LARGURA_RESTAURADA, ALTURA_RESTAURADA, True)
    except Exception:
        pass

def treinar(continuar=False):
    os.makedirs(PASTA_MODELOS, exist_ok=True)
    
    env = tmrl.get_environment()
    env = TmrlVisaoWrapper(env)
    
    print("\n[Treino] Iniciando IA com Visão Computacional...")
    
    modelo_path = MODELO_FINAL + ".zip"
    if continuar and os.path.exists(modelo_path):
        print(f"Retomando o modelo de: {modelo_path}")
        modelo = SAC.load(modelo_path, env=env)
    else:
        print("Criando novo modelo SAC MultiInput (CNN + MLP)...")
        # MultiInputPolicy é a chave para ler imagens e números ao mesmo tempo
        modelo = SAC(
            "MultiInputPolicy", 
            env, 
            verbose=1,
            learning_rate=3e-4, 
            buffer_size=50_000, # Reduzido para não estourar a RAM com imagens
            batch_size=256,
            tau=0.02,
            device="auto" # Manda o processamento pesado para a Placa de Vídeo
        )
        
    try:
        modelo.learn(total_timesteps=TOTAL_PASSOS, progress_bar=True)
    except KeyboardInterrupt:
        print("\nTreino parado pelo utilizador. Guardando progresso...")
        
    modelo.save(MODELO_FINAL)
    print(f"Modelo guardado em: {MODELO_FINAL}.zip")
    
    env.close()
    restaurar_janela()

def jogar(episodios=5):
    modelo_path = MODELO_FINAL + ".zip"
    if not os.path.exists(modelo_path):
        print("Modelo não encontrado. Treine primeiro.")
        return
        
    print(f"\n[Jogar] Avaliando modelo por {episodios} episódios...")
    env = tmrl.get_environment()
    env = TmrlVisaoWrapper(env)
    modelo = SAC.load(modelo_path, env=env)
    
    for ep in range(episodios):
        obs, _ = env.reset()
        done = False
        recompensa_total = 0
        
        while not done:
            acao, _ = modelo.predict(obs, deterministic=True)
            obs, recompensa, terminated, truncated, _ = env.step(acao)
            recompensa_total += recompensa
            done = terminated or truncated
            
        print(f"Episódio {ep+1} finalizado! Recompensa Final: {recompensa_total:.2f}")

    env.close()
    restaurar_janela()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "eval"], default="train")
    parser.add_argument("--continuar", action="store_true")
    args = parser.parse_args()

    if args.mode == "train":
        treinar(args.continuar)
    elif args.mode == "eval":
        jogar()