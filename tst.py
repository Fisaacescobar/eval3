

import numpy as np
import pandas as pd
from utility import act_sigmoid

np.random.seed(1)

def load_data():
    X = pd.read_csv("dtest.csv", header=None).values
    Y = pd.read_csv("classtest.csv", header=None).values
    return X, Y

def load_W():
    V1 = np.load("output/V1.npy")
    V2 = np.load("output/V2.npy")
    W  = np.load("output/W_soft.npy")
    return V1, V2, W

def zscores_data(X):
    m = X.mean(axis=0)
    s = X.std(axis=0)
    s[s == 0] = 1.0
    return (X - m) / s

def forward_softmax(X, V1, V2, W):
    Xd = X.T
    H1 = act_sigmoid(V1 @ Xd)
    H2 = act_sigmoid(V2.T @ H1).T

    m2 = H2.mean(axis=0)
    s2 = H2.std(axis=0)
    s2[s2 == 0] = 1.0
    xv = (H2 - m2) / s2

    Xe = np.hstack([xv, np.ones((xv.shape[0], 1))])
    z = Xe @ W
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    Yp = e / np.sum(e, axis=1, keepdims=True)
    return Yp

def measures(Y_true, Y_pred):
    y_true = np.argmax(Y_true, axis=1)
    y_pred = np.argmax(Y_pred, axis=1)
    K = Y_true.shape[1]

    cm = np.zeros((K, K), dtype=int)
    for i in range(len(y_true)):
        cm[y_true[i], y_pred[i]] += 1

    fs = []
    for k in range(K):
        TP = cm[k, k]
        FP = cm[:, k].sum() - TP
        FN = cm[k, :].sum() - TP
        p = TP / (TP + FP) if (TP + FP) > 0 else 0
        r = TP / (TP + FN) if (TP + FN) > 0 else 0
        f = 2*p*r/(p+r) if (p+r) > 0 else 0
        fs.append(f)

    return cm, fs, float(np.mean(fs))

def save_best(cm, fs):
    pd.DataFrame(cm).to_csv("confusion.csv", index=False, header=False)
    pd.DataFrame(fs).to_csv("fscores.csv", index=False, header=False)
    
    
def save_macro(cm_macro, f_macro):
    pd.DataFrame(cm_macro).to_csv("confusion_macro.csv", index=False, header=False)
    pd.DataFrame(f_macro).to_csv("fscores_macro.csv", index=False, header=False)



def main():		
    X, Y = load_data()
    V1, V2, W = load_W()
    
    RUNS = 5
    all_cm = []
    all_fs = []
    all_macro = []
    
    for r in range(RUNS):
    
        Xz = zscores_data(X)
        Yp = forward_softmax(Xz, V1, V2, W)
        cm, fs, macro = measures(Y, Yp)

        all_cm.append(cm)
        all_fs.append(fs)
        all_macro.append(macro)


    best_idx = int(np.argmax(all_macro))
    best_cm = all_cm[best_idx]
    best_fs = all_fs[best_idx] + [all_macro[best_idx]]

    print("Matriz de Confusión (BEST):")
    print(best_cm)
    print("F-score:", best_fs)

    save_best(best_cm, best_fs)

    cm_macro = sum(all_cm)
    f_macro_avg = [float(np.mean([fs[k] for fs in all_fs])) for k in range(Y.shape[1])]
    f_macro_total = float(np.mean(all_macro))
    f_macro_out = f_macro_avg + [f_macro_total]
    print("Matriz de Confusión (MACRO):")
    print(cm_macro)
    print("F-scores promedio por clase:", f_macro_avg)
    print("F-score macro total:", f_macro_total)
    
    save_macro(cm_macro, f_macro_out)
    


    print("\nArchivos guardados:")
    print(" → confusion.csv       (mejor resultado)")
    print(" → fscores.csv         (mejor resultado)")
    print(" → confusion_macro.csv (5 corridas sumadas)")
    print(" → fscores_macro.csv   (promedios)")

    print("[TST] Test completado")

    
if __name__ == '__main__':   
    main()