# Reaction-diffusion simulation of bacterial colonisation
# Explicit and semi-implicit finite-difference solvers
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

np.random.seed(42)  # make the initial perturbation reproducible

# setting the size of petri dish domain (10cm) , and defining the fixed parameters/ coefficients in the PDE
domain_size = 10
Du, Dv = 0.16, 0.05  # Diffusion coefficients (same for explicit and implicit methods)
F, k = 0.08, 0.01     # Glucose feed and bacterial decay rates
threshold = 0.1       # threshold for E. coli to be considered present
r_physical = 1        # Seed radius in cm
t_ref = 3600.0        # Reference time (seconds)
real_growth_time = 10 * 3600.0  # 10 hours in seconds
growth_time = real_growth_time / t_ref  #hours of growth simulated by code (10 for this coursework)

# defining N values- grid points per dimension x and y
N_values = [50, 100,150,200,250,300,350,400]

#initialise lists
coverage_exp = []  # stores Explicit final coverage
coverage_imp = []  # stores Implicit final coverage
time_exp = []      # stores Time points for explicit
coverage_exp_time = []  # stores Coverage over time for explicit (N=100 only)
v_exp_final = None  # Store final explicit bacterial coverage after 10 hours, at N=400, to plot heatmap


# Setting Boundary conditions (no-flux conditions applied through definition of laplacian at edges, see CW summary)
#Explicit Laplacian function: 
def laplacian(z, h):
    lap = np.zeros_like(z)
    #Neumann (no-flux) Boundary conditions:
    lap[1:-1, 1:-1] = (z[2:, 1:-1] + z[:-2, 1:-1] + z[1:-1, 2:] + z[1:-1, :-2] - 4 * z[1:-1, 1:-1]) / h**2      #interior points (five-point stencil)
    lap[0, 1:-1] = (2 * z[1, 1:-1] + z[0, 2:] + z[0, :-2] - 4 * z[0, 1:-1]) / h**2          #top
    lap[-1, 1:-1] = (2 * z[-2, 1:-1] + z[-1, 2:] + z[-1, :-2] - 4 * z[-1, 1:-1]) / h**2     #bottom
    lap[1:-1, 0] = (z[2:, 0] + z[:-2, 0] + 2 * z[1:-1, 1] - 4 * z[1:-1, 0]) / h**2      #left
    lap[1:-1, -1] = (z[2:, -1] + z[:-2, -1] + 2 * z[1:-1, -2] - 4 * z[1:-1, -1]) / h**2     #right
    lap[0, 0] = (2 * z[1, 0] + 2 * z[0, 1] - 4 * z[0, 0]) / h**2        #top-left corner
    lap[0, -1] = (2 * z[1, -1] + 2 * z[0, -2] - 4 * z[0, -1]) / h**2        #top-right corner
    lap[-1, 0] = (2 * z[-2, 0] + 2 * z[-1, 1] - 4 * z[-1, 0]) / h**2        #bottom-left corner
    lap[-1, -1] = (2 * z[-2, -1] + 2 * z[-1, -2] - 4 * z[-1, -1]) / h**2    #bottom-right corner
    return lap
 
#Semi-implicit matrix function: 
#defining implicit matrix (for diffusion terms, explained in CW summary)
def build_implicit_matrix(D, dt, h, N):
    #discretised diffusion operator alpha (scaling diffusion coefficient by dt/h**2 for stability and fair comparison between explicit and implicit)
    alpha=D*dt/(h**2)  
    main_diag = (1 + 4 * alpha) * np.ones(N * N)  #main diagonal of the matrix, coefficient of n+1th term
    off_diag = -alpha* np.ones(N * N - 1)
    off_diag[N-1: :N] = 0
    #vertical set up of the matrix
    diag_up_N = -alpha* np.ones(N * N - N)
    diag_down_N = -alpha * np.ones(N * N - N)
    #no-flux condition (missing neighbours are adjusted, laplacian is adjusted at edges):
    for i in range(N):
        main_diag[i] -= alpha
        main_diag[-i-1] -= alpha
    for i in range(0, N*N, N):
        main_diag[i] -= alpha
    for i in range(N-1, N*N, N):
        main_diag[i] -= alpha
    #matrix A
    A = diags([diag_down_N, off_diag, main_diag, off_diag, diag_up_N], [-N, -1, 0, 1, N], shape=(N*N, N*N), format='csr')
    return A

#defining grid spacing (h), and the region of petri grid where bacterial seed will be initialised 
for N in N_values:
    h = domain_size / N
    x, y = np.meshgrid(np.arange(N), np.arange(N))
    mask = (x - N//2)**2 + (y - N//2)**2 < (r_physical / h)**2      #mask for seed

    # In this section I am implementing the numerical method 
    #Explicit Method, forward-time, central-space 
    dt_exp = 1.5 * h**2         #time step for explicit (to satisfy courant condition, see section D of CW Summary)
    steps_exp = int(growth_time / dt_exp)
    #initialise conditions
    u_exp = np.ones((N, N)) + 0.01 * np.random.random((N, N))       #glucose concentration
    v_exp = np.zeros((N, N)) + 0.01 * np.random.random((N, N))      #E. coli concentration
    v_exp[mask] = 1  #E. coli concentration inside seed=1
    start_time = time.time()
    if N == 100:  # Collect time data for N=100
        time_exp = [0]
        coverage_exp_time = [np.sum(v_exp > threshold) / (N * N) * 100]

#explicit simulation
for n in range(steps_exp):
    Lu, Lv = laplacian(u_exp, h), laplacian(v_exp, h) 
    u_new = u_exp + dt_exp * (Du * Lu - u_exp * v_exp**2 + F * (1 - u_exp))         #current values of glucose concentration (updated using explicit Euler method)
    v_new = v_exp + dt_exp * (Dv * Lv + u_exp * v_exp**2 - (F + k) * v_exp)         #current values of E. coli concentration (updated using explicit Euler method)
    u_exp, v_exp = np.clip(u_new, 0, 1), np.clip(v_new, 0, 1)
    #% E.coli coverage over time for N=100
    if N == 100 and n % (steps_exp // 100) == 0:  # Sample every ~0.1 hour
        time_exp.append((n + 1) * dt_exp)
        coverage_exp_time.append(np.sum(v_exp > threshold) / (N * N) * 100)     #final %coverage
#final coverage
coverage_exp.append(np.sum(v_exp > threshold) / (N * N) * 100)
#final v for N=400 heatmap
if N == 400:  
    v_exp_final = v_exp.copy()
print(f"Explicit N = {N}, Coverage = {coverage_exp[-1]:.2f}%, Time = {time.time() - start_time:.2f} s")


#Implicit Method, backward-Euler implicit, forward-Euler explicit 
dt_imp = 0.1
steps_imp = int(growth_time / dt_imp)
#initialise glucose and E.coli concentration
u_imp = np.ones((N, N)) + 0.01 * np.random.random((N, N))
v_imp = np.zeros((N, N)) + 0.01 * np.random.random((N, N))
v_imp[mask] = 1  #centred seed
#build implicit matrices
Au = build_implicit_matrix(Du, dt_imp, h, N)
Av = build_implicit_matrix(Dv, dt_imp, h, N)
start_time = time.time()
#implicit simulation:
for n in range(steps_imp):
    #right-hand-side of the Gray-Scott Eqt, reaction terms
    rhs_u = u_imp + dt_imp * (-u_imp * v_imp**2 + F * (1 - u_imp))
    rhs_v = v_imp + dt_imp * (u_imp * v_imp**2 - (F + k) * v_imp)
    #solving implicit system
    u_new_flat = spsolve(Au, rhs_u.flatten())
    v_new_flat = spsolve(Av, rhs_v.flatten())
    u_imp = np.clip(u_new_flat.reshape(N, N), 0, 1)
    v_imp = np.clip(v_new_flat.reshape(N, N), 0, 1)
coverage_imp.append(np.sum(v_imp > threshold) / (N * N) * 100)  #final coverage
print(f"Implicit N = {N}, Coverage = {coverage_imp[-1]:.2f}%, Time = {time.time() - start_time:.2f} s")

 
# Plotting graphs
plt.figure(figsize=(18, 5))
 
# Plot 1: Coverage vs Nodes
nodes = [(N * N)/1000 for N in N_values]        #in thousand 
plt.subplot(1, 3, 1)
plt.plot(nodes, coverage_exp, 'bo-', label='Explicit')
plt.plot(nodes, coverage_imp, 'ro-', label='Implicit')
plt.xlabel('Number of Nodes (thousand)')
plt.ylabel('Percentage Coverage (%)')
plt.title('% baterial coverage over 10 hrs vs. Nodes ')
plt.grid(True)
plt.legend()
 
# Plot 2: Explicit Coverage vs Time (N=100)
plt.subplot(1, 3, 2)
plt.plot(time_exp, coverage_exp_time, 'b-', label='Explicit (N=100)')
plt.xlabel('Time (hours)')
plt.ylabel('Percentage Coverage (%)')
plt.title('Explicit % Coverage vs. Time ')
plt.grid(True)
plt.legend()
 
# Plot 3: Heatmap of Bacterial Density (N=400, Explicit)
plt.subplot(1, 3, 3)
plt.imshow(v_exp_final, cmap='inferno', extent=[0, domain_size, 0, domain_size])
plt.colorbar(label='Bacterial Density (v)')
plt.xlabel('x (cm)')
plt.ylabel('y (cm)')
plt.title('E.coli in 10 Hours (N=400, Explicit)')
 
plt.tight_layout() #prevent overlap
plt.show()
 
