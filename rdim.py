
from __future__ import annotations

import math
import os
import numpy as np
import pandas as pd


from utility import iniW, act_sigmoid

np.random.seed(1)


def w2_pinv(xb: np.ndarray, W1: np.ndarray, C: float) -> np.ndarray:
    H = act_sigmoid(W1 @ xb)  # (K1 × M)
    HHt = H @ H.T             # (K1 × K1)
    I = np.eye(HHt.shape[0], dtype=W1.dtype)
    U = (xb @ H.T) @ np.linalg.inv(HHt + I / C)  # (d × K1)
    return U


def forward(xb, W1, W2):
    H = act_sigmoid(W1 @ xb) # (K1 × M)
    Y = W2 @ H               # (d × M)
    return H, Y 
    

def gradV(H: np.ndarray, U: np.ndarray, xb: np.ndarray) -> np.ndarray:
    Y = U @ H                 # (d × M)
    err = Y - xb             # (d × M)
    g_hidden = (U.T @ err) * (H * (1.0 - H))  # (K1 × M)
    grad = (g_hidden @ xb.T) / xb.shape[1]  # (K1 × d)
    mse = float(np.mean(np.sum(err**2, axis=0)))  
    return grad, mse

def gradW1(H, W2, xb):
    grad, _ = gradV(H, W2, xb)
    return grad


def upd_adam(
    W: np.ndarray,
    m: np.ndarray,
    v: np.ndarray,
    grad: np.ndarray,
    mu: float,
    t: int,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    m = beta1 * m + (1.0 - beta1) * grad
    v = beta2 * v + (1.0 - beta2) * (grad * grad)
    m_hat = m / (1.0 - beta1 ** t)
    v_hat = v / (1.0 - beta2 ** t)
    W = W - mu * m_hat / (np.sqrt(v_hat) + eps)
    return W, m, v

def train_minibatch(Xe: np.ndarray,
                    W1: np.ndarray,
                    W2: np.ndarray,
                    m: np.ndarray,
                    v: np.ndarray,
                    mu: float,
                    C: float,
                    BatchSize: int,
                    t_init: int):

    N = Xe.shape[1]
    B = math.ceil(N / BatchSize)
    Cost = np.zeros(B)
    t = t_init

    for n in range(B):
        start = n * BatchSize
        end = min((n + 1) * BatchSize, N)
        xb = Xe[:, start:end]    # (d × M)

        # Etapa #1 — W2
        W2 = w2_pinv(xb, W1, C)

        # Etapa #2 — gradiente y Adam
        H, Y = forward(xb, W1, W2)
        gW1, Cost[n] = gradV(H, W2, xb)
        t += 1
        W1, m, v = upd_adam(W1, m, v, gW1, mu, t)

    return float(np.mean(Cost)), W1, W2, m, v, t



def train_sls(X: np.ndarray, K1: int, C: float, MaxIter: int, BatchSize: int, mu: float):

    X = zscores_dataset(X)
    
    Xd = X.T
    d, N = Xd.shape


    W1 = iniW(K1, d)      # (K1 × d)
    W2 = np.zeros((d, K1))
    m = np.zeros_like(W1)
    v = np.zeros_like(W1)
    t = 0

    for Iter in range(1, MaxIter + 1):

        # Desordenar dataset
        idx = np.random.permutation(N)
        Xe = Xd[:, idx]

        mse_avg, W1, W2, m, v, t = train_minibatch(
            Xe, W1, W2, m, v, mu, C, BatchSize, t
        )

        if Iter % 10 == 0:
            print(f"[SLS] Iter {Iter}/{MaxIter} — MSE = {mse_avg:.6f}")

    return W1



def pc_svd(H: np.ndarray, K2: int) -> tuple[np.ndarray, np.ndarray]:
    Hc = H - H.mean(axis=1, keepdims=True)
    N = H.shape[1]
    Y = Hc.T / math.sqrt(max(1, N - 1))  # (N × K1)

    U, S_vals, Vt = np.linalg.svd(Y, full_matrices=False)
    V = Vt.T  # (K1 × K1)
    V2 = V[:, :K2]  # (K1 × K2)
    return V2, S_vals



def load_dataset(train_path: str) -> np.ndarray:
    
    data = pd.read_csv(train_path, header=None)
    X = data.values
    
    return X


def load_param(config_path: str) -> dict:
    config = pd.read_csv(config_path, header=None)
    C = float(config.iloc[0, 0])
    max_iter = int(config.iloc[1, 0])
    batch_size = int(config.iloc[2, 0])
    mu = float(config.iloc[3, 0])

    row = str(config.iloc[4, 0])
    parts = row.replace(',', ' ').split()
    K1 = int(parts[0])
    K2 = int(parts[1]) if len(parts) > 1 else int(config.iloc[5, 0])

    return {
        'C': C,
        'max_iter': max_iter,
        'batch_size': batch_size,
        'mu': mu,
        'K1': K1,
        'K2': K2
    }


def save_new_data(X_reduced: np.ndarray, V1: np.ndarray, V2: np.ndarray, Vr: dict, output_dir: str = 'output') -> None:
    print(f"[SAVE] Guardando resultados en directorio: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, 'X_reduced.npy'), X_reduced)
    np.save(os.path.join(output_dir, 'V1.npy'), V1)
    np.save(os.path.join(output_dir, 'V2.npy'), V2)
    np.save(os.path.join(output_dir, 'Vr.npy'), Vr)
    print("[SAVE] Datos guardados: X_reduced.npy, V1.npy, V2.npy, Vr.npy")


def zscores_dataset(X: np.ndarray) -> np.ndarray:
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X_norm = (X - mean) / std
    return X_norm
    


def main() -> None:
    # Crear diccionario Vr {V1 , V2}    
    Vr = {}
    X = load_dataset('dtrain.csv')
    
    params = load_param('config_sls.csv')
    K1=params['K1']; K2=params['K2']; C=params['C']; max_iter=params['max_iter']; batch_size=params['batch_size']; mu=params['mu']
    

    # Reducir dimensionalidad
    X_norm = zscores_dataset(X)
    
    V1 = train_sls(X_norm, K1, C, max_iter, batch_size, mu)
    Vr[1] = V1

    H_all = act_sigmoid(V1 @ X_norm.T)  # (K1 × N)
    
    V2, S_vals = pc_svd(H_all, K2)
    Vr[2] = V2
    
    energy = np.cumsum(S_vals**2) / np.sum(S_vals**2)
    print("[RDIM] Energía acumulada de valores singulares:")
    for j in range(min(len(S_vals), 10)):
        print(f"     Componente {j+1}: S²={S_vals[j]**2:.4f}, Acumulado={energy[j]:.4f}")
    
    
    Z = V2.T @ H_all  # (K2 × N)
    Z_act = act_sigmoid(Z)  # (K2 × N)
    X_reduced = Z_act.T  # (N × K2)
    
    # Guardar resultados
    save_new_data(X_reduced, V1, V2, Vr, output_dir='output')
    print("[MAIN] Proceso de reducción completado")


if __name__ == '__main__':
    main()