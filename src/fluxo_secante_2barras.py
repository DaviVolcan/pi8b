# =====================================================================
#  FLUXO DE POTENCIA EM SISTEMA DE 2 BARRAS - METODO DA SECANTE
#  Projeto Integrador VII-B - Engenharia Eletrica UCPel
#
#  Como executar:  python3 fluxo_secante_2barras.py
#  Requisitos:     numpy, matplotlib   (pip install numpy matplotlib)
#
#  TOPOLOGIA
#  Barra 1 (slack): V1 = 1,0 pu, theta1 = 0 (referencia angular)
#  Barra 2 (PQ)   : carga (Pload, Qload) + geracao FV opcional (Pgd)
#  Linha 1-2      : Z = R + jX [pu]
#
#  MODELO MATEMATICO
#  Do balanco de potencia na barra 2, S2 = V2 . conj(I), eliminando o
#  angulo theta2, chega-se a equacao biquadratica na magnitude V = |V2|:
#
#    f(V) = V^4 + [ 2(P.R + Q.X) - V1^2 ].V^2 + (P^2+Q^2)(R^2+X^2) = 0
#
#  com P, Q = potencias liquidas CONSUMIDAS na barra 2 (P = Pload - Pgd).
#  A raiz fisica e obtida pelo METODO DA SECANTE:
#
#    V[k+1] = V[k] - f(V[k]) . (V[k] - V[k-1]) / (f(V[k]) - f(V[k-1]))
#
#  Angulo recuperado por: theta2 = atan2(Q.R - P.X, V^2 + P.R + Q.X)
# =====================================================================

import numpy as np
import matplotlib.pyplot as plt

# --------------------------- DADOS [pu] -----------------------------
V1    = 1.00     # tensao da barra slack
R     = 0.05     # resistencia da linha
X     = 0.10     # reatancia da linha
Pload = 0.50     # carga ativa da barra 2
Qload = 0.25     # carga reativa da barra 2
Pgd   = 0.00     # geracao fotovoltaica na barra 2 (FP unitario)

tol   = 1e-10    # tolerancia de convergencia
kmax  = 50       # numero maximo de iteracoes


# ------------------- FUNCOES DE FLUXO DE POTENCIA -------------------
def f(V, P, Q):
    return V**4 + (2*(P*R + Q*X) - V1**2)*V**2 + (P**2 + Q**2)*(R**2 + X**2)

def df(V, P, Q):                # derivada analitica (so para o Newton)
    return 4*V**3 + 2*(2*(P*R + Q*X) - V1**2)*V


# ------------------------ METODO DA SECANTE -------------------------
def secante(P, Q, x0, x1, tol=tol, kmax=kmax, verbose=False):
    err = []
    x2 = x1
    for k in range(1, kmax + 1):
        f0, f1 = f(x0, P, Q), f(x1, P, Q)
        d = f1 - f0
        if abs(d) < 1e-300:
            break
        x2 = x1 - f1*(x1 - x0)/d            # formula da secante
        e = abs(x2 - x1)
        err.append(e)
        if verbose:
            print(f"  {k:2d}   {x2:14.10f}   {f(x2,P,Q):12.4e}   {e:10.3e}")
        x0, x1 = x1, x2
        if e < tol:
            break
    return x2, np.array(err)


# ---------- NEWTON-RAPHSON (metodo de referencia p/ comparar) -------
def newton(P, Q, x, tol=tol, kmax=kmax, verbose=False):
    err = []
    for k in range(1, kmax + 1):
        dx = -f(x, P, Q)/df(x, P, Q)
        x += dx
        e = abs(dx)
        err.append(e)
        if verbose:
            print(f"  {k:2d}   {x:14.10f}   {f(x,P,Q):12.4e}   {e:10.3e}")
        if e < tol:
            break
    return x, np.array(err)


# ===================================================================
#  PARTE 1 - CASO BASE: processo iterativo e velocidade de convergencia
# ===================================================================
P, Q = Pload - Pgd, Qload

print("\n================= METODO DA SECANTE =================")
print("   k       V2 [pu]           f(V2)         |dV|")
V2, err_sec = secante(P, Q, 0.90, 1.00, verbose=True)

print("\n============= NEWTON-RAPHSON (referencia) ===========")
print("   k       V2 [pu]           f(V2)         |dV|")
V2nr, err_nr = newton(P, Q, 1.00, verbose=True)

theta2 = np.arctan2(Q*R - P*X, V2**2 + P*R + Q*X)   # [rad]
V2c    = V2*np.exp(1j*theta2)
I      = (V1 - V2c)/(R + 1j*X)
Ploss  = R*abs(I)**2

# validacao: raiz analitica da biquadratica (Bhaskara em V^2)
b = 2*(P*R + Q*X) - V1**2
c = (P**2 + Q**2)*(R**2 + X**2)
Va = np.sqrt((-b + np.sqrt(b**2 - 4*c))/2)

print("\n--------------------- RESULTADOS --------------------")
print(f" V2 (secante)   = {V2:.10f} pu")
print(f" V2 (analitico) = {Va:.10f} pu  -> desvio = {abs(V2-Va):.2e}")
print(f" theta2         = {np.degrees(theta2):.4f} graus")
print(f" |I| na linha   = {abs(I):.4f} pu")
print(f" Perdas ativas  = {Ploss:.5f} pu ({100*Ploss/Pload:.2f} % da carga)")
print(f" Iteracoes: secante = {len(err_sec)}  |  Newton-Raphson = {len(err_nr)}")

plt.figure(1)
plt.semilogy(range(1, len(err_sec)+1), np.maximum(err_sec, 1e-16), '-ob', label="Secante")
plt.semilogy(range(1, len(err_nr)+1),  np.maximum(err_nr,  1e-16), '-sr', label="Newton-Raphson")
plt.grid(True, which="both")
plt.xlabel("Iteracao k")
plt.ylabel("Erro |V(k+1) - V(k)|  [pu]")
plt.title("Convergencia: Secante (ordem ~1,618) x Newton-Raphson (ordem 2)")
plt.legend()

# ===================================================================
#  PARTE 2 (bonus) - Varredura de GD: ligacao com o Hosting Capacity
# ===================================================================
Pgd_v = np.arange(0, 3.0 + 1e-9, 0.05)
V2_v  = np.zeros_like(Pgd_v)
v0, v1 = 0.90, 1.00
for n, pg in enumerate(Pgd_v):
    Pn = Pload - pg                       # P < 0 => fluxo reverso
    Vn, _ = secante(Pn, Qload, v0, v1)
    V2_v[n] = Vn
    v0, v1 = Vn - 0.02, Vn + 0.02         # warm start

HC = Pgd_v[V2_v <= 1.05][-1]
print(f"\n Hosting Capacity (limite 1,05 pu): Pgd_max = {HC:.2f} pu")

plt.figure(2)
plt.plot(Pgd_v, V2_v, '-b', label="V2 (fluxo resolvido pela secante)")
plt.axhline(1.05, color='r', ls='--', label="Limite PRODIST 1,05 pu")
plt.grid(True)
plt.xlabel("Geracao FV na barra 2, Pgd [pu]")
plt.ylabel("Tensao V2 [pu]")
plt.title("Elevacao de tensao com a penetracao de GD - Hosting Capacity")
plt.legend()

plt.show()
