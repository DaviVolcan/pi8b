# =====================================================================
#  FLUXO DE POTENCIA - EXEMPLO 6.8 (Glover & Sarma)
#  Sistema de 3 barras com barra PV - METODO DE BROYDEN
#  Projeto Integrador VII-B - Engenharia Eletrica UCPel
#
#  Como executar:  python3 exemplo_6_8.py
#  Requisitos:     numpy, matplotlib
#
#  TOPOLOGIA (malha completa, base 100 MVA)
#    Barra 1 (slack): V1 = 1,05 pu, theta1 = 0
#    Barra 2 (PQ)   : Pcarga = 400 MW, Qcarga = 250 Mvar
#    Barra 3 (PV)   : Pger   = 200 MW, |V3| = 1,04 pu (fixo)
#    Linha 1-2: Z = 0,02 + j0,04 pu
#    Linha 1-3: Z = 0,01 + j0,03 pu
#    Linha 2-3: Z = 0,0125 + j0,025 pu
#    Susceptancias shunt desprezadas
#
#  VARIAVEIS DE ESTADO: x = [theta2, V2, theta3]   (3 incognitas)
#    Bus PQ (2): theta2 e V2 livres  ->  equacoes DP2, DQ2
#    Bus PV (3): theta3 livre, |V3| fixo  ->  equacao DP3 apenas
#  RESIDUOS: F = [DP2, DQ2, DP3]
#
#  BROYDEN (rank-1):
#    dx = -J^{-1} F(x)
#    x  <- x + dx
#    J  <- J + (DF - J dx) dx^T / (dx^T dx)
# =====================================================================

import numpy as np
import matplotlib.pyplot as plt

# ---- dados [pu, base 100 MVA] ----
V1     = 1.05          # slack
V3_mag = 1.04          # magnitude fixada (barra PV)

P2_esp = -400 / 100    # -4.0 pu  (carga)
Q2_esp = -250 / 100    # -2.5 pu  (carga)
P3_esp = +200 / 100    # +2.0 pu  (geracao)

tol  = 1e-10
kmax = 50

# ---- Ybus (malha completa: linhas 1-2, 1-3, 2-3) ----
y12 = 1.0 / (0.02   + 1j*0.04)
y13 = 1.0 / (0.01   + 1j*0.03)
y23 = 1.0 / (0.0125 + 1j*0.025)

Ybus = np.zeros((3, 3), dtype=complex)
Ybus[0, 0] =  y12 + y13
Ybus[1, 1] =  y12 + y23
Ybus[2, 2] =  y13 + y23
Ybus[0, 1] = Ybus[1, 0] = -y12
Ybus[0, 2] = Ybus[2, 0] = -y13
Ybus[1, 2] = Ybus[2, 1] = -y23
G, B = Ybus.real, Ybus.imag


# ---- residuos: x = [theta2, V2, theta3] ----
def mismatch(x):
    th2, V2, th3 = x
    V  = [V1, V2, V3_mag]
    th = [0.0, th2, th3]

    P2 = sum(V[1]*V[j]*(G[1,j]*np.cos(th[1]-th[j]) + B[1,j]*np.sin(th[1]-th[j])) for j in range(3))
    Q2 = sum(V[1]*V[j]*(G[1,j]*np.sin(th[1]-th[j]) - B[1,j]*np.cos(th[1]-th[j])) for j in range(3))
    P3 = sum(V[2]*V[j]*(G[2,j]*np.cos(th[2]-th[j]) + B[2,j]*np.sin(th[2]-th[j])) for j in range(3))

    return np.array([P2_esp - P2, Q2_esp - Q2, P3_esp - P3])


def jac_num(x, h=1e-7):
    F0 = mismatch(x)
    J  = np.zeros((3, 3))
    for j in range(3):
        xp = x.copy(); xp[j] += h
        J[:, j] = (mismatch(xp) - F0) / h
    return J


# =====================================================================
#  METODO DE BROYDEN
# =====================================================================
def broyden(x0, tol=tol, kmax=kmax, verbose=False):
    x = x0.copy()
    F = mismatch(x)
    J = jac_num(x)
    err = []
    for k in range(1, kmax + 1):
        dx    = np.linalg.solve(J, -F)
        x_new = x + dx
        F_new = mismatch(x_new)
        e = np.linalg.norm(dx)
        err.append(e)
        if verbose:
            print(f"  {k:2d}   th2={np.degrees(x_new[0]):8.4f}  V2={x_new[1]:.6f}"
                  f"  th3={np.degrees(x_new[2]):8.4f}  ||dx||={e:.3e}")
        if e < tol:
            x = x_new
            break
        dF = F_new - F
        J  = J + np.outer(dF - J @ dx, dx) / (dx @ dx)
        x, F = x_new, F_new
    return x, np.array(err)


# =====================================================================
#  NEWTON-RAPHSON (referencia)
# =====================================================================
def newton_pf(x0, tol=tol, kmax=kmax, verbose=False):
    x = x0.copy()
    err = []
    for k in range(1, kmax + 1):
        dx = np.linalg.solve(jac_num(x), -mismatch(x))
        x += dx
        e  = np.linalg.norm(dx)
        err.append(e)
        if verbose:
            print(f"  {k:2d}   th2={np.degrees(x[0]):8.4f}  V2={x[1]:.6f}"
                  f"  th3={np.degrees(x[2]):8.4f}  ||dx||={e:.3e}")
        if e < tol:
            break
    return x, np.array(err)


# =====================================================================
#  CASO BASE
# =====================================================================
x0 = np.array([0.0, 1.0, 0.0])   # partida plana

print("\n======= BROYDEN - EXEMPLO 6.8 =======")
print("   k    theta2[deg]    V2[pu]    theta3[deg]   ||dx||")
xB,  err_B  = broyden(x0, verbose=True)

print("\n======= NEWTON-RAPHSON (referencia) =======")
print("   k    theta2[deg]    V2[pu]    theta3[deg]   ||dx||")
xNR, err_NR = newton_pf(x0, verbose=True)

th2, V2, th3 = xB
V  = [V1, V2, V3_mag]
th = [0.0, th2, th3]

# potencias na barra slack e Q3 (barra PV)
P1 = sum(V[0]*V[j]*(G[0,j]*np.cos(th[0]-th[j]) + B[0,j]*np.sin(th[0]-th[j])) for j in range(3))
Q1 = sum(V[0]*V[j]*(G[0,j]*np.sin(th[0]-th[j]) - B[0,j]*np.cos(th[0]-th[j])) for j in range(3))
Q3 = sum(V[2]*V[j]*(G[2,j]*np.sin(th[2]-th[j]) - B[2,j]*np.cos(th[2]-th[j])) for j in range(3))

print("\n--------------------- RESULTADOS (Broyden) --------------------")
print(f" V1 = {V1:.4f} pu   theta1 =  0.0000 graus  (slack)")
print(f" V2 = {V2:.6f} pu   theta2 = {np.degrees(th2):.4f} graus  (PQ)")
print(f" V3 = {V3_mag:.4f} pu   theta3 = {np.degrees(th3):.4f} graus  (PV)")
print(f"\n P1 = {P1*100:.2f} MW    Q1 = {Q1*100:.2f} Mvar  (slack)")
print(f" P3 = {P3_esp*100:.2f} MW    Q3 = {Q3*100:.2f} Mvar  (PV)")
print(f" P2 = {P2_esp*100:.2f} MW    Q2 = {Q2_esp*100:.2f} Mvar  (PQ, carga)")

# fluxos e perdas nas linhas
print("\n---- Fluxos e Perdas nas Linhas ----")
lines = [(0, 1, y12), (0, 2, y13), (1, 2, y23)]
Ploss_total = 0.0
for (i, j, y) in lines:
    Vi  = V[i] * np.exp(1j*th[i])
    Vj  = V[j] * np.exp(1j*th[j])
    Iij = (Vi - Vj) * y
    Sij = Vi * np.conj(Iij)          # fluxo de i para j
    Sji = Vj * np.conj(-Iij)         # fluxo de j para i
    Pl  = (Sij + Sji).real
    Ploss_total += Pl
    print(f" Linha {i+1}-{j+1}: "
          f"S{i+1}{j+1} = {Sij.real*100:+7.2f} MW {Sij.imag*100:+7.2f} Mvar | "
          f"S{j+1}{i+1} = {Sji.real*100:+7.2f} MW {Sji.imag*100:+7.2f} Mvar | "
          f"Perdas = {Pl*100:.3f} MW")

print(f"\n Perdas totais = {Ploss_total*100:.3f} MW")
print(f"\n Iteracoes: Broyden = {len(err_B)}   Newton-Raphson = {len(err_NR)}")
print(f" ||F|| Broyden        = {np.linalg.norm(mismatch(xB)):.2e}")
print(f" ||F|| Newton-Raphson = {np.linalg.norm(mismatch(xNR)):.2e}")

plt.figure(1)
plt.semilogy(range(1, len(err_B)+1),  np.maximum(err_B,  1e-16), '-ob', label="Broyden")
plt.semilogy(range(1, len(err_NR)+1), np.maximum(err_NR, 1e-16), '-sr', label="Newton-Raphson")
plt.grid(True, which='both')
plt.xlabel("Iteracao k")
plt.ylabel("||dx||  [pu]")
plt.title("Convergencia: Broyden x Newton-Raphson (Exemplo 6.8)")
plt.legend()

plt.show()