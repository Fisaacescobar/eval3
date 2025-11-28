

import json
import math
import numpy as np
import pandas as pd

np.random.seed(1)

def updW_adam(
    W: np.ndarray,
    V: np.ndarray,
    S: np.ndarray,
    grad: np.ndarray,
    mu: float,
    t: int,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    V = beta1 * V + (1.0 - beta1) * grad
    S = beta2 * S + (1.0 - beta2) * (grad * grad)
    V_hat = V / (1.0 - beta1 ** t)
    S_hat = S / (1.0 - beta2 ** t)
    W = W - mu * V_hat / (np.sqrt(S_hat) + eps)
    return W, V, S


def softmax_grad(y_pred: np.ndarray, xe: np.ndarray, ye: np.ndarray) -> tuple[np.ndarray, float]:
    grad = (xe.T @ (y_pred - ye)) / xe.shape[0]
    cost = -float(np.mean(np.sum(ye * np.log(y_pred + 1e-15), axis=1)))
    return grad, cost

def train_minibatch(Xe, Ye, W, V, S, mu, BatchSize, t0):
    N = Xe.shape[0]
    B = math.ceil(N / BatchSize)
    Cost = np.zeros(B)
    t = t0

    for n in range(B):
        s = n * BatchSize
        e = min((n + 1) * BatchSize, N)
        xb = Xe[s:e]
        yb = Ye[s:e]

        z = xb @ W
        z = z - np.max(z, axis=1, keepdims=True)
        eZ = np.exp(z)
        Yp = eZ / np.sum(eZ, axis=1, keepdims=True)

        gW, Cost[n] = softmax_grad(Yp, xb, yb)

        t += 1
        W, V, S = updW_adam(W, V, S, gW, mu, t)

    return float(np.mean(Cost)), W, V, S, t

def train_softmax(X, Y, MaxIter, BatchSize, mu):
    N, d = X.shape
    K = Y.shape[1]

    Xe = np.hstack([X, np.ones((N, 1))])
    D = Xe.shape[1]

    W = np.random.randn(D, K) * 0.01
    V = np.zeros_like(W)
    S = np.zeros_like(W)

    CostHist = np.zeros(MaxIter)
    t = 0

    for Iter in range(1, MaxIter + 1):

        perm = np.random.permutation(N)
        Xe_s = Xe[perm]
        Ye_s = Y[perm]

        CostHist[Iter - 1], W, V, S, t = train_minibatch(
            Xe_s, Ye_s, W, V, S, mu, BatchSize, t
        )

        if Iter % 50 == 0:
            print(f"[TRN] Iter {Iter}/{MaxIter} — Coste = {CostHist[Iter-1]:.6f}")

    return W, CostHist


def zscores_data(X):
    m = X.mean(axis=0)
    s = X.std(axis=0)
    s[s == 0] = 1
    return (X - m) / s





def load_param( config_softmax_path: str) -> dict:
    print(f"[LOAD PARAM] Cargando parámetros desde: {config_softmax_path}")
    # Cargar parámetros de Softmax
    config_softmax = pd.read_csv(config_softmax_path, header=None)
    max_iter_softmax = int(config_softmax.iloc[0, 0])
    batch_size_softmax = int(config_softmax.iloc[1, 0])
    mu_softmax = float(config_softmax.iloc[2, 0])

    params = {

        "Softmax": {
            "max_iter": max_iter_softmax,
            "batch_size": batch_size_softmax,
            "mu": mu_softmax
        }
    }

    return params 


    
    
    


def main() -> None:
    print("[TRN] Inicio del entrenamiento Softmax")

    X = np.load('output/X_reduced.npy')
    Y = pd.read_csv('classtrain.csv', header=None).values

    params = load_param('config_softmax.csv')
    softmax_params = params['Softmax']
    mu_soft = softmax_params['mu']; max_iter = softmax_params['max_iter'] ; batch_size = softmax_params['batch_size']
    
    X = zscores_data(X)
    
    print(f"[TRN] Parámetros Softmax: mu_soft={mu_soft}, max_iter={max_iter}, batch_size={batch_size}")
    W_soft, costs = train_softmax(X, Y, max_iter, batch_size, mu_soft)
    
    print("[TRN] Entrenamiento finalizado. Coste final medio:", costs[-1])
    # Guardar pesos
    
    np.save('output/W_soft.npy', W_soft)
    print("[TRN] Pesos W_soft guardados en W_soft.npy")
    np.savetxt("costos.csv",costs, delimiter="," )


if __name__ == '__main__':
    main()


