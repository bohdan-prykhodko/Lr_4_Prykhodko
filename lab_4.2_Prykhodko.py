import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

# --------------------------
# 1. ОГОЛОШЕННЯ ЗМІННИХ
# --------------------------
x, y = sp.symbols('x y')
r, phi = sp.symbols('r phi', positive=True)

a, b, c, d = sp.symbols('a b c d')

# --------------------------
# 2. ЗАДАЄМО ЛІНІЙНУ СИСТЕМУ
#     x' = ax + by
#     y' = cx + dy
# --------------------------
P = a*x + b*y
Q = c*x + d*y

print("P(x,y) =", P)
print("Q(x,y) =", Q)

# --------------------------
# 3. ПЕРЕХІД ДО ПОЛЯРНИХ КООРДИНАТ:
#     x = r cos φ
#     y = r sin φ
# --------------------------
Pp = P.subs({x: r*sp.cos(phi), y: r*sp.sin(phi)})
Qp = Q.subs({x: r*sp.cos(phi), y: r*sp.sin(phi)})

print("\nP(r,φ) =", Pp)
print("Q(r,φ) =", Qp)

# --------------------------
# 4. ВИРАХОВУЄМО r' і φ' (формули з методички)
# --------------------------
r_dot = (a*sp.cos(phi) + b*sp.sin(phi))*sp.cos(phi) \
        + (c*sp.cos(phi) + d*sp.sin(phi))*sp.sin(phi)

phi_dot = (a*sp.cos(phi) + b*sp.sin(phi))*sp.sin(phi) \
        - (c*sp.cos(phi) + d*sp.sin(phi))*sp.cos(phi)

r_dot = sp.simplify(r_dot)
phi_dot = sp.simplify(phi_dot)

print("\n r' =", r_dot)
print(" φ' =", phi_dot)

# --------------------------
# 5. dr/dφ = r'/φ'
# --------------------------
dr_dphi = sp.simplify(r_dot / phi_dot)
print("\n dr/dφ =", dr_dphi)

# --------------------------
# 6. ПОЛІНОМ N(t) для напрямів:
#     N(tan φ) = 0
# формула з методички:
#     b * tan^2 φ + (a - d) tan φ - c = 0
# --------------------------
t = sp.symbols('t')  # t = tan φ

N_poly = b*t**2 + (a - d)*t - c
print("\nN(tan φ) =", N_poly)

# --------------------------
# 7. Розв'язуємо N(t)=0 → напрями φ_k
# --------------------------
# ---- ПОШУК φ ДЛЯ N(tan φ)=0 ----

a_val = 1
b_val = 2
c_val = -3
d_val = 1

N_poly_num = N_poly.subs({a:a_val,b:b_val,c:c_val,d:d_val})

phi_solutions = sp.solve(N_poly_num, t)

phi_angles = []

for sol in phi_solutions:
    if sol.is_real:
        sol_float = float(sol)
        angle = np.arctan(sol_float)
        phi_angles.append(angle)
    else:
        print("Комплексний корінь → напрям пропускається:", sol)

print("Дійсні напрямки φ =", phi_angles)


# --------------------------
# 8. ХАРАКТЕРИСТИЧНЕ РІВНЯННЯ
# --------------------------
lambda_symbol = sp.symbols('lambda')
char_eq = sp.factor(lambda_symbol**2 - (a + d)*lambda_symbol + (a*d - b*c))
print("\nХарактеристичне рівняння =", char_eq)

lam = sp.solve(char_eq, lambda_symbol)
print("Корені λ₁,₂ =", lam)

# --------------------------
# 9. УМОВА ПЕРІОДИЧНОСТІ (відсутність дійсних коренів)
#     (a - d)^2 + 4bc < 0
# --------------------------
periodic_cond = sp.simplify((a - d)**2 + 4*b*c)
print("\nУмова періодичності (має бути < 0):", periodic_cond)

# --------------------------
# 10. ПОБУДОВА ФАЗОВОГО ПОРТРЕТУ
# --------------------------
a_num = 1
b_num = 2
c_num = -3
d_num = 1

P_num = sp.lambdify((x, y), P.subs({a: a_num, b: b_num, c: c_num, d: d_num}), 'numpy')
Q_num = sp.lambdify((x, y), Q.subs({a: a_num, b: b_num, c: c_num, d: d_num}), 'numpy')

X, Y = np.meshgrid(np.linspace(-5, 5, 25), np.linspace(-5, 5, 25))
u = P_num(X, Y)
v = Q_num(X, Y)

plt.figure(figsize=(8, 8))
plt.quiver(X, Y, u, v, color='blue', alpha=0.6)

# Лінії напрямів
for ang in phi_angles:
    t = np.linspace(-5, 5, 200)
    plt.plot(t*np.cos(ang), t*np.sin(ang), 'r', linewidth=2)

plt.axhline(0, color='black', linewidth=0.7)
plt.axvline(0, color='black', linewidth=0.7)

plt.title("Фазовий портрет лінійної системи та напрями N(tan φ)=0")
plt.xlabel("x")
plt.ylabel("y")
plt.axis('equal')
plt.grid(True)
plt.show()
