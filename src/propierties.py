import numpy as np
import json
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Global Settings
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 20
rcParams['legend.fontsize'] = 20
rcParams['mathtext.fontset'] = 'cm' 
rcParams['axes.labelsize'] = 18

# Global Constants
R  = 8.314      # Universal Gas Constant [J/(mol K)]
T0 = 273.2      # Reference Temperature of water [K]
Lo = 333.6      # Latent heat of fusion for water [kJ/kg]

def load_composition(filename: str) -> dict:
    """Loads composition parameters from a JSON file."""
    try:
        with open(filename, 'r') as file:
            params = json.load(file)
        return params
    except Exception as e:
        raise RuntimeError(f"Error loading {filename}: {e}")

def calculate_T_cong(xs: float) -> float:
    """Calculates initial freezing point depression via empirical polynomial."""
    return -32.523 * xs**2 - 0.6781 * xs

def compute_ice_and_water(T: np.ndarray, T_cong: float, humedad: float) -> (np.ndarray, np.ndarray):
    """Calculates temperature-dependent mass fractions of ice and unfrozen water."""
    x_hielo = np.where(T < T_cong, humedad * (1 - (T_cong / T)), 0)
    x_agua = humedad - x_hielo
    return x_hielo, x_agua

def compute_density(T: np.ndarray, composicion: np.ndarray) -> np.ndarray:
    """Calculates mixture density assuming ideal volume additivity."""
    # Empirical polynomials for pure component densities (kg/m^3)
    rho_protein       = 1.3299e3 - 5.1840e-1 * T
    rho_lipidos       = 9.2559e2 - 4.1757e-1 * T
    rho_carbohidratos = 1.5991e3 - 3.1046e-1 * T
    rho_fibra         = 1.3115e3 - 3.6589e-1 * T
    rho_cenizas       = 2.4238e3 - 2.8063e-1 * T
    rho_agua          = 9.9718e2 + 3.1439e-3 * T - 3.7574e-3 * T**2
    rho_hielo         = 9.1689e2 - 1.3071e-1 * T

    rho_i = np.vstack([rho_protein, rho_lipidos, rho_carbohidratos, rho_fibra,
                       rho_cenizas, rho_agua, rho_hielo])
    
    # Apply inverse mass fraction rule: rho_mix = 1 / sum(x_i / rho_i)
    rho_extracto = 1 / np.sum(composicion / rho_i, axis=0)
    return rho_extracto

def compute_cp(T: np.ndarray, composicion: np.ndarray, xs: float, humedad: float, proteinas: float, T_cong: float) -> np.ndarray:
    """Calculates apparent specific heat capacity accounting for latent heat of fusion."""
    # Bound water fraction assumption
    xb = 0.4 * proteinas
    
    # Effective molecular weight of soluble solids via Clausius-Clapeyron relation
    Ms = -(xs * R * T0**2) / ((humedad - xb) * Lo * T_cong)
    
    # Empirical polynomials for pure component specific heats (kJ/(kg K))
    cp_protein       = 2.0082 + 1.2089e-3 * T - 1.3129e-6 * T**2
    cp_lipidos       = 1.9842 + 1.4733e-3 * T - 4.8008e-6 * T**2
    cp_carbohidratos = 1.5488 + 1.9625e-3 * T - 5.9399e-6 * T**2
    cp_fibra         = 1.8459 + 1.8306e-3 * T - 4.4609e-6 * T**2
    cp_cenizas       = 1.0926 + 1.8896e-3 * T - 3.6817e-6 * T**2
    cp_hielo         = 2.0623 + 6.0769e-3 * T
    cp_agua          = np.where(T <= 0,
                              4.1289 - 5.3062e-3 * T + 9.9516e-4 * T**2,
                              4.1289 - 9.0864e-5 * T + 5.4731e-6 * T**2)
    
    cp_matrix = np.vstack([cp_protein, cp_lipidos, cp_carbohidratos, cp_fibra,
                           cp_cenizas, cp_agua, cp_hielo])
    
    cp_extracto = np.zeros_like(T)
    mask = T < T_cong
    
    # Schwartzberg apparent specific heat model for frozen state (incorporates df_ice/dT)
    cp_extracto[mask] = 1.55 + 1.26 * xs + (xs * R * T0**2) / (Ms * T[mask]**2)
    
    # Mass-weighted average for unfrozen state
    cp_extracto[~mask] = np.sum(composicion[:, ~mask] * cp_matrix[:, ~mask], axis=0)
    return cp_extracto

def compute_conductivity(T: np.ndarray) -> np.ndarray:
    """Calculates pure component thermal conductivities."""
    k_protein       = 1.7881e-1 + 1.1958e-3 * T - 2.7178e-6 * T**2
    k_lipidos       = 1.8071e-1 - 2.7604e-4 * T - 1.7749e-7 * T**2
    k_carbohidratos = 2.0141e-1 + 1.3874e-3 * T - 4.3312e-6 * T**2
    k_fibra         = 1.8331e-1 + 1.2497e-3 * T - 3.1683e-6 * T**2
    k_cenizas       = 3.2962e-1 + 1.4011e-3 * T - 2.9069e-6 * T**2
    k_agua          = 5.7109e-1 + 1.7625e-3 * T - 6.7036e-6 * T**2
    k_hielo         = 2.2196    - 6.2489e-3 * T + 1.0154e-4 * T**2

    return np.vstack([k_protein, k_lipidos, k_carbohidratos, k_fibra,
                      k_cenizas, k_agua, k_hielo])

def calculate_xk(sustancias: dict) -> dict:
    """Calculates effective thermal conductivity using an iterative Maxwell-Eucken model."""
    sustancia_base = 'agua'
    rho_base = sustancias[sustancia_base]['rho']
    k_base = sustancias[sustancia_base]['k']
    x_base = sustancias[sustancia_base]['content']
    
    results = {}
    
    # Initialize cumulative volume fraction
    x_total = x_base / rho_base
    k_previous = k_base

    for sustancia, props in sustancias.items():
        if sustancia == sustancia_base:
            continue
        rho_current = props['rho']
        k_current = props['k']
        content_current = props['content']
        
        # Convert mass fraction to volume fraction
        x_current = content_current / rho_current
        x_total += x_current
        x_factor = x_current / x_total
        
        # Two-phase Maxwell-Eucken relation for dispersed/continuous medium
        k_combined = k_previous * (k_current + 2*k_previous - 2*x_factor*(k_previous - k_current)) / \
                     (k_current + 2*k_previous + x_factor*(k_previous - k_current))
        
        k_previous = k_combined
        results[sustancia] = {'x': x_factor, 'k': k_combined}
    return results

def plot_properties(T: np.ndarray, rho_extract: np.ndarray, cp_extract: np.ndarray, k_extract: np.ndarray):
    """Generates a 1x3 subplot figure for density, heat capacity, and thermal conductivity."""
    plt.figure(figsize=(18, 5))

    # Density
    plt.subplot(1, 3, 1)
    plt.plot(T, rho_extract, '--o', label=r"Density", markevery=10)
    plt.xlabel(r'Temperature [$^{\circ}$C]')
    plt.ylabel(r'[$kg/m^3$]', fontsize=32)
    plt.legend(loc='upper left')
    plt.grid(True, which="both", ls="-", alpha=0.3)

    # Heat Capacity
    plt.subplot(1, 3, 2)
    plt.plot(T, cp_extract , '--o', label='Heat\n capacity', markevery=10)
    plt.xlabel(r'Temperature [$^{\circ}$C]')
    plt.ylabel(r'[$kJ/(kg \cdot ^{\circ}$C)]', fontsize=32)
    plt.legend(loc='upper left')
    plt.grid(True, which="both", ls="-", alpha=0.3)

    # Thermal Conductivity
    plt.subplot(1, 3, 3)
    plt.plot(T, k_extract, '--o', label='Thermal\n conductivity', markevery=10)
    plt.xlabel(r'Temperature [$^{\circ}$C]')
    plt.ylabel(r'[$W/(m \cdot ^{\circ}$C)]', fontsize=32)
    plt.legend(loc='upper right')
    plt.grid(True, which="both", ls="-", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("propiertiesPaper.pdf", format="pdf", bbox_inches="tight")
    plt.show()

# ==========================================
# Execution Pipeline
# ==========================================

# Load mass fractions from file
params = load_composition('compo.txt')
xs            = params['xs']
humedad       = params['humedad']
fibra         = params['fibra']
lipidos       = params['lipidos']
cenizas       = params['cenizas']
proteinas     = params['proteinas']
carbohidratos = params['carbohidratos']

# Temperature discretization
T = np.linspace(-20, 20, 100)
T_cong = calculate_T_cong(xs)

x_hielo, x_agua = compute_ice_and_water(T, T_cong, humedad)

# Assemble composition matrix (Shape: 7 x len(T))
composicion = np.vstack([
    proteinas      * np.ones_like(T),
    lipidos        * np.ones_like(T),
    carbohidratos  * np.ones_like(T),
    fibra          * np.ones_like(T),
    cenizas        * np.ones_like(T),
    x_agua,
    x_hielo
])

rho_extracto = compute_density(T, composicion)
cp_extracto = compute_cp(T, composicion, xs, humedad, proteinas, T_cong)

# Compute thermal conductivity iteratively
sustancias = {
    "agua": {
        "rho": 9.9718e2 + 3.1439e-3 * T - 3.7574e-3 * T**2,
        "k": 5.7109e-1 + 1.7625e-3 * T - 6.7036e-6 * T**2,
        "content": x_agua
    },
    "fibra": {
        "rho": 1.3115e3 - 3.6589e-1 * T,
        "k": 1.8331e-1 + 1.2497e-3 * T - 3.1683e-6 * T**2,
        "content": fibra
    },
    "lipidos": {
        "rho": 9.2559e2 - 4.1757e-1 * T,
        "k": 1.8071e-1 - 2.7604e-4 * T - 1.7749e-7 * T**2,
        "content": lipidos
    },
    "cenizas": {
        "rho": 2.4238e3 - 2.8063e-1 * T,
        "k": 3.2962e-1 + 1.4011e-3 * T - 2.9069e-6 * T**2,
        "content": cenizas
    },
    "proteinas": {
        "rho": 1.3299e3 - 5.1840e-1 * T,
        "k": 1.7881e-1 + 1.1958e-3 * T - 2.7178e-6 * T**2,
        "content": proteinas
    },
    "carbohidratos": {
        "rho": 1.5991e3 - 3.1046e-1 * T,
        "k": 2.0141e-1 + 1.3874e-3 * T - 4.3312e-6 * T**2,
        "content": carbohidratos
    },
    "hielo": {
        "rho": 9.1689e2 - 1.3071e-1 * T,
        "k": 2.2196 - 6.2489e-3 * T + 1.0154e-4 * T**2,
        "content": x_hielo
    }
}

resultados = calculate_xk(sustancias)
# Extract final effective conductivity after all components are iteratively added
k_extracto = resultados['hielo']['k']

plot_properties(T, rho_extracto, cp_extracto, k_extracto)

k_0 = max(k_extracto)
rho_0 = max(rho_extracto)
cp_0 = max(cp_extracto)

Temp_ref = 20.0

# Calculate property curves
C_nondim_curve = (rho_extracto * (cp_extracto))/(rho_0 * cp_0)
K_nondim_curve = k_extracto / k_0

# Guardar los valores en un archivo CSV
data = np.column_stack((T/Temp_ref, C_nondim_curve, K_nondim_curve))
np.savetxt("properties_dataNonDim.csv", data, delimiter=",", header="T_params, C_nondim_curve, K_nondim_curve", comments="")
