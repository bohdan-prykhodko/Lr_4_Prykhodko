import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------
# 1. ОГОЛОШЕННЯ ЗМІННИХ
# ------------------------------
x, y = sp.symbols('x y', real=True)
r, phi = sp.symbols('r phi', real=True, positive=True)

# ------------------------------
# 2. ЗАДАЄМО ОДНОРІДНУ СИСТЕМУ
#    ẋ = P(x, y),   ẏ = Q(x, y)
# ------------------------------
# ТУТ ВСТАВЛЯЄШ СВОЇ P(x,y) та Q(x,y),
# ВОНИ МАЮТЬ БУТИ ОДНОРІДНИМИ ОДНОГО СТЕПЕНЯ.

# ПРИКЛАД: система 2-го степеня
#   ẋ = x^2 - y^2
#   ẏ = 2xy
P = x**2 - y**2
Q = 2*x*y

print("P(x,y) =", P)
print("Q(x,y) =", Q)

# Перевіримо ступінь однорідності
degP = sp.total_degree(P)
degQ = sp.total_degree(Q)
print("deg P =", degP, " deg Q =", degQ)

m = degP  # припускаємо, що degP = degQ


# ------------------------------
# 3. ПЕРЕХІД ДО ПОЛЯРНИХ КООРДИНАТ
#    x = r cos φ,  y = r sin φ
# ------------------------------
P_polar = sp.simplify(P.subs({x: r*sp.cos(phi), y: r*sp.sin(phi)}))
Q_polar = sp.simplify(Q.subs({x: r*sp.cos(phi), y: r*sp.sin(phi)}))

print("\nP(r,φ) =", P_polar)
print("Q(r,φ) =", Q_polar)

# За теорією для однорідних:
# P(r,φ) = r^m * B(φ)
# Q(r,φ) = r^m * A(φ)

B_phi = sp.simplify(P_polar / r**m)
A_phi = sp.simplify(Q_polar / r**m)

print("\nA(φ) =", A_phi)
print("B(φ) =", B_phi)

# ------------------------------
# 4. ОБЧИСЛЮЄМО Z(φ), N(φ)
#    Z(φ) = A(φ) sin φ + B(φ) cos φ
#    N(φ) = -A(φ) cos φ + B(φ) sin φ
# ------------------------------
Z_phi = sp.simplify(A_phi*sp.sin(phi) + B_phi*sp.cos(phi))
N_phi = sp.simplify(-A_phi*sp.cos(phi) + B_phi*sp.sin(phi))

print("\nZ(φ) =", Z_phi)
print("N(φ) =", N_phi)

# ------------------------------
# 5. РІВНЯННЯ dr/dφ = r * Z(φ) / N(φ)
# ------------------------------
dr_dphi = sp.simplify(r * Z_phi / N_phi)
print("\nРівняння: dr/dφ =", dr_dphi)


# ------------------------------
# 6. НАПРЯМКИ ПРЯМИХ ЧЕРЕЗ ОСОБЛИВІ ТОЧКИ:
#    N(φ) = 0  -> φ_k
# ------------------------------
# Шукаємо корені N(φ)=0 на [0, 2π)

phi_roots = []

for guess in np.linspace(0, 2*np.pi, 8, endpoint=False):
    try:
        root = sp.nsolve(N_phi, guess)
        root_mod = float(root % (2*sp.pi))
        # уникаємо дублювання коренів
        if all(abs(root_mod - r0) > 1e-3 for r0 in phi_roots):
            phi_roots.append(root_mod)
    except (ValueError, sp.SympifyError):
        pass

phi_roots = sorted(phi_roots)
print("\nНапрямки φ_k (N(φ_k)=0):")
for r0 in phi_roots:
    print("φ ≈", r0)

# ------------------------------
# 7. ВІЗУАЛІЗАЦІЯ: ПРЯМІ ЧЕРЕЗ ОСОБЛИВІ ТОЧКИ
# ------------------------------
# Намалюємо кілька променів з початку координат під кутами φ_k,
# а також коротке фазове поле для наочності.

# Сітка для векторного поля
Xn, Yn = np.meshgrid(np.linspace(-2, 2, 25), np.linspace(-2, 2, 25))
Pn = sp.lambdify((x, y), P, 'numpy')(Xn, Yn)
Qn = sp.lambdify((x, y), Q, 'numpy')(Xn, Yn)

plt.figure(figsize=(7, 7))
plt.quiver(Xn, Yn, Pn, Qn, alpha=0.4)

# Рисуємо промені φ_k
for r0 in phi_roots:
    t = np.linspace(0, 2, 100)
    xs = t * np.cos(r0)
    ys = t * np.sin(r0)
    plt.plot(xs, ys, 'r', linewidth=2)

plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)

plt.title("Однорідна система в околі (0,0) та напрями N(φ)=0")
plt.xlabel("x")
plt.ylabel("y")
plt.axis('equal')
plt.grid(True)
plt.show()
