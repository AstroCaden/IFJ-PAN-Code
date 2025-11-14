import os
import shutil
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from scipy.integrate import solve_ivp
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

#Setup

max_eos = None           # None: Uses all EoS files, otherwise will generate up to that number of EoS.
EoS_Generations = 500          # Number of stars generated for each EoS.
Min_Pressure = 1e-3            # Minimum pressure in MeV/fm^3.

# Paths for each input and output file
Directory      = os.getcwd()
Hadronic_EoS   = os.path.join(Directory, 'EoS_DD2MEV_p0_SLy4.dat')
Quark_Base     = os.path.join(Directory, 'Quark_EoS')

Hybrid_Base    = os.path.join(Directory, 'Generated_Hybrid_EoS_FINAL')
PMu_Output     = os.path.join(Directory, 'Individual_PMu_Plots')
PEpsi_Output   = os.path.join(Directory, 'Individual_PEpsilon_Plots')
Indiv_MR_Output = os.path.join(Directory, 'Individual_MR_Plots')
Indiv_LM_Output = os.path.join(Directory, 'Individual_LM_Plots')
Stellar_Output = os.path.join(Directory, 'Star_Data_Output')

#Constants in cgs and convsertions to that.

G  = 6.67430e-8
C  = 2.99792458e10
M_Sun = 1.98847e33
Km_To_Cm = 1.0e5

Pressure_Geometric = G / C**4
Density_Geometric  = G / C**2

MeVfm3_to_dynecm2 = 1.60218e33
MeVfm3_to_Gcm3    = 1.78266e12

#Folder creater and cleaner

def empty_folder(folder_path, create_if_not_exists=True):
    """Empties folders and creates them if they are not present."""
    if os.path.exists(folder_path):
        for name in os.listdir(folder_path):
            path = os.path.join(folder_path, name)
            if os.path.isfile(path) or os.path.islink(path):
                os.unlink(path)
            elif os.path.isdir(path):
                shutil.rmtree(path)
    elif create_if_not_exists:
        os.makedirs(folder_path, exist_ok=True)
    print(f"\n Clean {folder_path}")

#Deletes any previous outputs

print("\n Cleaning output folders")
empty_folder(Hybrid_Base)
empty_folder(PMu_Output)
empty_folder(PEpsi_Output)
empty_folder(Indiv_MR_Output)
empty_folder(Indiv_LM_Output)
empty_folder(Stellar_Output)

#Creates names for final output plots.

main_plot_files = [
    'Mass_vs_Radius_FINAL.png',
    'Lambda_vs_Mass_FINAL.png',
    'Pressure_vs_Mu_Overall_FINAL.png',
    'Pressure_vs_Epsilon_Overall_FINAL.png',
]
for plot_file in main_plot_files:
    path = os.path.join(Directory, plot_file)
    if os.path.exists(path):
        os.remove(path)
        print(f"  Remove {plot_file}")

print("\nCleanup complete")

#Part 1: Uses the Phase Transition seen as the crossover in the Pressure-Chemical Potential plots.

print("\nPart 1: Building hybrid EoS.")

# Load and sort hadronic EoS
hadronic_data = np.loadtxt(Hadronic_EoS)
hadronic_data = hadronic_data[np.argsort(hadronic_data[:, 0])]

mu_h = hadronic_data[:, 0]
p_h  = hadronic_data[:, 2]   # Pressure vs Chemical Potential in Hadronic EoS.

p_H_interp = interp1d(mu_h, p_h, kind='linear', bounds_error=False, fill_value="extrapolate")

# Determines quark EoS paths.
quark_filepaths = sorted(
    os.path.join(root, f)
    for root, _, files in os.walk(Quark_Base)
    for f in files
    if f.endswith('.dat') and not f.startswith('._')
)

print(f"Found {len(quark_filepaths)} quark EoS files.")
print(f"Max hybrid EoS to generate: {max_eos or 'all'}")

Hybrid_EoS_Paths = []    # Paths to generated hybrid EoS files.
PMu_Data = []            # For combined Pressure-Chemical Potential plot.
PEpsi_Data = []          # For combined Pressure-Energy Density plots.
Phase_Trans_Prop = {}    # Stores Chemical Potential, Pressure, etc. at Phase Transition

files_done = 0

for filepath in quark_filepaths:
    if max_eos is not None and files_done >= max_eos:
        print(f"\nReached max_eos = {max_eos}. Stopping EoS generation.")
        break

    eos_basename    = os.path.basename(filepath)
    parent_dir_name = os.path.basename(os.path.dirname(filepath))
    eos_label = f"{parent_dir_name}_{eos_basename}"   # unique label
    hybrid_label = f"Hybrid_{eos_label}"

    print(f"\nProcessing quark EoS: {eos_label}")

    # Loads and stores all quark EoS.
    quark_data = np.loadtxt(filepath)
    quark_data = quark_data[np.argsort(quark_data[:, 0])]

    mu_q = quark_data[:, 0]
    p_q  = quark_data[:, 1]

    p_Q_interp = interp1d(mu_q, p_q, kind='linear', bounds_error=False, fill_value="extrapolate")

    # Bounded interpolators for intersection search
    p_H_bound = interp1d(mu_h, p_h, kind='linear', bounds_error=True)
    p_Q_bound = interp1d(mu_q, p_q, kind='linear', bounds_error=True)

    # Determines crossover in the Pressure-Chemical Potential plot for Phase Transition.
    if np.any(p_h > 0) and np.any(p_q > 0):
        mu_min = max(mu_h[p_h > 0].min(), mu_q[p_q > 0].min())
        mu_max = min(mu_h.max(),           mu_q.max())
    else:
        mu_min, mu_max = None, None

    mu_c = None
    p_trans = None

    if mu_min is not None and mu_min < mu_max:
        # Looks for sign change in Hadronic and Quark pressures to find the Phase Transition.
        test_mu = np.linspace(mu_min, mu_max, 1000)
        diff = p_H_bound(test_mu) - p_Q_bound(test_mu)
        sign_change = np.where(np.diff(np.sign(diff)))[0]

        if sign_change.size > 0:
            a = test_mu[sign_change[0]]
            b = test_mu[sign_change[0] + 1]
            mu_c = brentq(lambda x: p_H_bound(x) - p_Q_bound(x), a, b)
            p_trans = p_H_bound(mu_c)
            if p_trans < Min_Pressure:
                mu_c = None
                p_trans = None

    if mu_c is None:
        # If there is no Phase Transition, save the plot and move on.
        print("  No Phase Transition found above Min_Pressure. Skipping.")

        mask_h = p_h >= Min_Pressure
        mask_q = p_q >= Min_Pressure

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(mu_h[mask_h], p_h[mask_h], 'b--', alpha=0.7, label='Hadronic')
        ax.plot(mu_q[mask_q], p_q[mask_q], 'r--', alpha=0.7, label='Quark')
        ax.set(
            xlabel=r'Chemical potential $\mu$ [MeV]',
            ylabel=r'Pressure $P$ [MeV/fm$^3$]',
            title=f'P–μ for {eos_label} (no transition)',
            xscale='log', yscale='log'
        )
        ax.grid(True, which='both', linestyle='--')
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(PMu_Output, f'P_vs_Mu_{eos_label}_NoTransition.png'))
        plt.close(fig)
        continue

    # Builds hybrid EoS, below mu_c it is Hadronic, above it is Quark matter.
    # Smooths slightly to prevent issues.
    n_h = np.gradient(p_H_interp(mu_h), mu_h)
    eps_h = mu_h * n_h - p_h
    eps_H_of_mu = interp1d(mu_h, eps_h, kind='linear', fill_value="extrapolate")
    eps_h_trans = float(eps_H_of_mu(mu_c))

    n_q = np.gradient(p_Q_interp(mu_q), mu_q)
    eps_q = mu_q * n_q - p_q
    eps_Q_of_mu = interp1d(mu_q, eps_q, kind='linear', fill_value="extrapolate")
    eps_q_trans = float(eps_Q_of_mu(mu_c))

    Phase_Trans_Prop[hybrid_label] = {
        'mu_c': mu_c,
        'p_trans': p_trans,
        'e_trans_hadronic': eps_h_trans,
        'e_trans_quark': eps_q_trans,
        'delta_epsilon': eps_q_trans - eps_h_trans
    }

    # Build dense grid in mu space for hybrid EoS.
    mu_dense_low  = np.geomspace(mu_h.min(), mu_c, 500, endpoint=False)
    mu_dense_high = np.geomspace(mu_c,      max(mu_q.max(), mu_c*1.0001), 500)
    mu_dense = np.sort(np.unique(np.concatenate([mu_dense_low, mu_dense_high])))

    p_dense_hybrid = np.where(mu_dense < mu_c, p_H_interp(mu_dense), p_Q_interp(mu_dense))
    p_smooth = gaussian_filter1d(p_dense_hybrid, sigma=2.0)

    n_dense = np.gradient(p_smooth, mu_dense)
    eps_dense = mu_dense * n_dense - p_smooth

    # Remove unphysical regions.
    valid = (
        (eps_dense > 0.0) &
        (p_smooth  > 0.0) &
        (np.gradient(eps_dense) > 0.0) &
        (p_smooth >= Min_Pressure)
    )

    if np.sum(valid) < 10:
        print("  Hybrid EoS has too few valid points after filtering. Skipping.")
        continue

    mu_valid  = mu_dense[valid]
    p_valid   = p_smooth[valid]
    eps_valid = eps_dense[valid]

    # Converts to cgs.
    rho_cgs = eps_valid * MeVfm3_to_Gcm3
    p_cgs   = p_valid * MeVfm3_to_dynecm2
    n_b     = n_dense[valid]

    out_path = os.path.join(Hybrid_Base, hybrid_label)
    np.savetxt(
        out_path,
        np.column_stack([rho_cgs, p_cgs, n_b]),
        fmt='%.6e',
        header="rho[g/cm^3]       p[dyn/cm^2]       n_b[fm^-3]"
    )

    Hybrid_EoS_Paths.append(out_path)
    files_done += 1
    print(f"  Hybrid EoS saved: {hybrid_label}  (success count: {files_done})")

    # Store data for combined plots.
    PMu_Data.append({'mu': mu_valid,   'p': p_valid})
    PEpsi_Data.append({'eps': eps_valid, 'p': p_valid})

    # Plots Pressure-Chemical Potential individually.
    mask_h_plot = p_h >= Min_Pressure
    mask_q_plot = p_q >= Min_Pressure

    fig_pmu, ax_pmu = plt.subplots(figsize=(8, 6))
    ax_pmu.plot(mu_h[mask_h_plot], p_h[mask_h_plot], 'b--', alpha=0.7, label='Hadronic')
    ax_pmu.plot(mu_q[mask_q_plot], p_q[mask_q_plot], 'r--', alpha=0.7, label='Quark')
    ax_pmu.plot(mu_valid, p_valid, 'g-', linewidth=2, label='Hybrid')
    ax_pmu.plot(mu_c, p_trans, 'ko', markersize=7,
                label=rf'$\mu_c$={mu_c:.2f} MeV, P={p_trans:.2f} MeV/fm$^3$')

    ax_pmu.set(
        xlabel=r'Chemical potential $\mu$ [MeV]',
        ylabel=r'Pressure $P$ [MeV/fm$^3$]',
        title=f'Hybrid EoS P–μ: {eos_label}',
        xscale='log', yscale='log'
    )
    ax_pmu.grid(True, which='both', linestyle='--')
    ax_pmu.legend()
    fig_pmu.tight_layout()
    fig_pmu.savefig(os.path.join(PMu_Output, f'P_vs_Mu_{eos_label}.png'))
    plt.close(fig_pmu)

    # Plots Pressure-Energy Density individually.
    fig_pe, ax_pe = plt.subplots(figsize=(8, 6))
    had_mask = mu_valid < mu_c
    q_mask   = ~had_mask

    if np.any(had_mask):
        ax_pe.plot(eps_valid[had_mask], p_valid[had_mask], 'r-', label='Hadronic branch')
    if np.any(q_mask):
        ax_pe.plot(eps_valid[q_mask],   p_valid[q_mask],   'b-', label='Quark branch')

    ax_pe.set(
        xlabel=r'Energy density $\epsilon$ [MeV/fm$^3$]',
        ylabel=r'Pressure $P$ [MeV/fm$^3$]',
        title=f'Hybrid EoS P–ε: {eos_label}',
        xscale='log', yscale='log'
    )
    ax_pe.grid(True, which='both', linestyle='--')
    ax_pe.legend()
    fig_pe.tight_layout()
    fig_pe.savefig(os.path.join(PEpsi_Output, f'P_vs_Epsilon_{eos_label}.png'))
    plt.close(fig_pe)

# Builds and plots combined Pressure-Chemical Potential and Pressure-Energy Density plots for all hybrid EoS.
fig_pmu_all, ax_pmu_all = plt.subplots(figsize=(10, 8))
fig_pe_all,  ax_pe_all  = plt.subplots(figsize=(10, 8))

for eos in PMu_Data:
    mask = eos['p'] >= Min_Pressure
    ax_pmu_all.plot(eos['mu'][mask], eos['p'][mask], alpha=0.7)

for eos in PEpsi_Data:
    mask = eos['p'] >= Min_Pressure
    ax_pe_all.plot(eos['eps'][mask], eos['p'][mask], alpha=0.7)

ax_pmu_all.set(
    xlabel=r'Chemical potential $\mu$ [MeV]',
    ylabel=r'Pressure $P$ [MeV/fm$^3$]',
    title='Hybrid EoS: combined P–μ',
    xscale='log', yscale='log'
)
ax_pmu_all.grid(True, which='both', linestyle='--')
fig_pmu_all.tight_layout()
fig_pmu_all.savefig('Pressure_vs_Mu_Overall_FINAL.png')

ax_pe_all.set(
    xlabel=r'Energy density $\epsilon$ [MeV/fm$^3$]',
    ylabel=r'Pressure $P$ [MeV/fm$^3$]',
    title='Hybrid EoS: combined P–ε',
    xscale='log', yscale='log'
)
ax_pe_all.grid(True, which='both', linestyle='--')
fig_pe_all.tight_layout()
fig_pe_all.savefig('Pressure_vs_Epsilon_Overall_FINAL.png')

plt.close(fig_pmu_all)
plt.close(fig_pe_all)

print(f"\nPart 1 complete: {len(Hybrid_EoS_Paths)} hybrid EoS generated.")

#Part 2: Builds the TOV solver and calculates Tidal Deformability.

print("\n Part 2: Solving TOV equations for each hybrid EoS")

def tov_equations(r, y, eps_of_p):
    """Right-hand side of TOV + tidal equations: y = [m(r), p(r), y_t(r)]."""
    m, p, y_t = y

    if p <= 0 or m < 0 or r <= 0:
        return [0.0, 0.0, 0.0]

    eps = eps_of_p(p)
    if eps <= 0:
        return [0.0, 0.0, 0.0]

    dmdr = 4.0 * np.pi * r**2 * eps

    if 2.0 * m >= r:
        return [0.0, 0.0, 0.0]

    dpdr = -(eps + p) * (m + 4.0 * np.pi * r**3 * p) / (r * (r - 2.0 * m))

    # Tidal equation terms, y_tidal
    e_2lambda = 1.0 / (1.0 - 2.0 * m / r)
    nu_prime  = 2.0 * (m + 4.0 * np.pi * r**3 * p) / (r * (r - 2.0 * m))

    #Approximates dP/DEpsilon in a finite difference to obtain c_s^2.
    dp_deps = 0.0
    dp_step = p * 1e-6 if p > 0 else 1e-20
    eps_plus = eps_of_p(p + dp_step)
    deps = eps_plus - eps
    if abs(deps) > 1e-20:
        dp_deps = dp_step / deps

    cs2_term = 0.0
    if dp_deps > 1e-20:
        cs2_term = (eps + p) / dp_deps

    F_r = 1.0 - 4.0 * np.pi * r**2 * (eps - p)
    Q_r = 4.0 * np.pi * r**2 * (5.0 * eps + 9.0 * p + cs2_term) - 6.0 * e_2lambda - (nu_prime * r)**2

    dy_t_dr = (-y_t**2 - y_t * F_r - Q_r) / r

    if not (np.isfinite(dmdr) and np.isfinite(dpdr) and np.isfinite(dy_t_dr)):
        return [0.0, 0.0, 0.0]

    return [dmdr, dpdr, dy_t_dr]

def surface_event(r, y, eps_of_p):
    """Stops integrating once the surface of the neutron star has been reached at its surface."""
    return y[1] - (1e-12 * Pressure_Geometric)

surface_event.terminal = True

tov_results = {} 

if EoS_Generations > 0 and Hybrid_EoS_Paths:
    for eos_path in Hybrid_EoS_Paths:
        eos_name = os.path.basename(eos_path)
        print(f"\n[TOV] Solving for: {eos_name}")
        
        # Loads the hybrid EoS for density and pressure, both in cgs.
        rho_cgs, p_cgs, _ = np.loadtxt(eos_path).T

        eps_geom = rho_cgs * Density_Geometric
        p_geom   = p_cgs   * Pressure_Geometric

        # Ensures that the pressure is monotonic.
        order = np.argsort(p_geom)
        p_geom   = p_geom[order]
        eps_geom = eps_geom[order]

        # Deletes any douplicates in pressure.
        unique_indices = np.where(np.diff(p_geom) > 0)[0] + 1
        p_unique   = np.concatenate(([p_geom[0]],   p_geom[unique_indices]))
        eps_unique = np.concatenate(([eps_geom[0]], eps_geom[unique_indices]))

        if p_unique.size < 2:
            print("  Not enough unique pressure points. Skipping.")
            continue

        eps_of_p = interp1d(p_unique, eps_unique, kind='linear', fill_value="extrapolate")

        # Central pressure range.
        p_c_min = max(p_unique.min(), 1e-20 * Pressure_Geometric)
        p_c_max = p_unique[-1] * 0.99
        if p_c_min >= p_c_max:
            print("  Invalid central pressure range. Skipping.")
            continue

        central_pressures = np.logspace(np.log10(p_c_min), np.log10(p_c_max), EoS_Generations)
        star_list = []

        for p_c in reversed(central_pressures):
            eps_c = eps_of_p(p_c)
            if eps_c <= 0 or not np.isfinite(eps_c):
                continue

            r0 = 1e-5
            m0 = (4.0 / 3.0) * np.pi * r0**3 * eps_c

            sol = solve_ivp(
                tov_equations,
                [r0, 50.0 * Km_To_Cm],
                [m0, p_c, 2.0],
                args=(eps_of_p,),
                events=surface_event,
                method='Radau',
                atol=1e-8,
                rtol=1e-8
            )

            if sol.status != 1 or sol.t_events[0].size == 0:
                continue

            R   = sol.t_events[0][0]
            M   = sol.y_events[0][0][0]
            y_R = sol.y_events[0][0][2]

            if R <= 0 or M <= 0 or R > 15.0 * Km_To_Cm:
                continue

            C_compact = M / R
            log_arg = 1.0 - 2.0 * C_compact
            if not (0 < C_compact < 0.5) or log_arg <= 0:
                continue

            # Calculates the Tidal Love number k2 and Lambda.
            num = (8.0 / 5.0) * C_compact**5 * (1.0 - 2.0 * C_compact)**2 * (
                2.0 + 2.0 * C_compact * (y_R - 1.0) - y_R
            )
            den = (
                2.0 * C_compact * (6.0 - 3.0 * y_R + 3.0 * C_compact * (5.0 * y_R - 8.0))
                + 4.0 * C_compact**3 * (
                    13.0 - 11.0 * y_R + C_compact * (3.0 * y_R - 2.0) + 2.0 * C_compact**2 * (1.0 + y_R)
                )
                + 3.0 * (1.0 - 2.0 * C_compact)**2 * (
                    2.0 - y_R + 2.0 * C_compact * (y_R - 1.0)
                ) * np.log(log_arg)
            )

            if abs(den) < 1e-20:
                continue

            k2 = num / den
            Lambda = (2.0 / 3.0) * k2 / C_compact**5
            if Lambda <= 0 or not np.isfinite(Lambda):
                continue

            # Conversions.
            M_solar = (M * C**2 / G) / M_Sun
            R_km    = R / Km_To_Cm

            # Conversions.
            p_c_mev   = p_c   * C**4 / G / MeVfm3_to_dynecm2
            eps_c_mev = eps_c * C**2 / G / MeVfm3_to_Gcm3

            star_list.append([M_solar, R_km, Lambda, p_c_mev, eps_c_mev])

        if not star_list:
            print("  No valid star models found for this EoS.")
            continue

        star_array = np.array(star_list)
        max_mass_idx = np.argmax(star_array[:, 0])
        p_c_at_max_mass = star_array[max_mass_idx, 3]

        tov_results[eos_path] = {
            'data': star_array,
            'p_c_at_max_mass': p_c_at_max_mass
        }

        print(f"  Found {len(star_array)} star solutions.")

        # Writes a table containing important properties of each star in seperate files.
        eos_stiffness = star_array[:, 0].max()
        trans_props = Phase_Trans_Prop.get(os.path.basename(eos_path), {})

        out_name = eos_name.replace('Hybrid_', '').replace('.dat', '.txt')
        out_path = os.path.join(Stellar_Output, out_name)

        with open(out_path, 'w') as f:
            f.write(f'# EoS file: {eos_name}\n')
            f.write(f'# Max mass (stiffness proxy): {eos_stiffness:.4f}\n')
            f.write(f"# mu_c: {trans_props.get('mu_c', 0.0):.4f} MeV\n")
            f.write(f"# P_trans: {trans_props.get('p_trans', 0.0):.4f} MeV/fm^3\n")
            f.write(f"# e_trans_hadronic: {trans_props.get('e_trans_hadronic', 0.0):.4f} MeV/fm^3\n")
            f.write(f"# e_trans_quark: {trans_props.get('e_trans_quark', 0.0):.4f} MeV/fm^3\n")
            f.write(f"# delta_epsilon: {trans_props.get('delta_epsilon', 0.0):.4f} MeV/fm^3\n")
            f.write(
                '# {0:<10}{1:<15}{2:<15}{3:<15}{4:<20}{5:<20}{6:<15}\n'.format(
                    "STAR_NUM", "MASS_SOLAR", "RADIUS_KM", "LAMBDA",
                    "CENTRAL_P_MEV", "CENTRAL_EPS_MEV", "STABILITY"
                )
            )

            for i, (M_solar, R_km, Lambda, p_c_mev, eps_c_mev) in enumerate(star_array):
                stability = "STABLE" if p_c_mev <= p_c_at_max_mass else "UNSTABLE"
                f.write(
                    f'  {i+1:<10}{M_solar:<15.6f}{R_km:<15.6f}{Lambda:<15.6f}'
                    f'{p_c_mev:<20.6f}{eps_c_mev:<20.6f}{stability:<15}\n'
                )

        print(f"  Saved star table: {out_name}")

        # Individual Mass-Radius and Tidal Deformability–Mass plots.
        if star_array.shape[0] < 2:
            print("  Not enough points for M–R / Λ–M plots.")
            continue

        M_solar_arr, R_km_arr, Lambda_arr, p_c_mev_arr, _ = star_array.T
        plot_base = eos_name.replace('Hybrid_', '')

        unstable_mask = p_c_mev_arr > p_c_at_max_mass
        stable_mask   = ~unstable_mask

        # M–R plot
        fig_mr_ind, ax_mr_ind = plt.subplots(figsize=(8, 6))
        ax_mr_ind.plot(R_km_arr, M_solar_arr, ':', color='grey', alpha=0.6, label='Full sequence')
        if np.any(stable_mask):
            ax_mr_ind.plot(R_km_arr[stable_mask], M_solar_arr[stable_mask],
                           'm.-', label='Stable')
        if np.any(unstable_mask):
            ax_mr_ind.plot(R_km_arr[unstable_mask], M_solar_arr[unstable_mask],
                           'r.--', label='Unstable')
        if max_mass_idx < len(M_solar_arr):
            ax_mr_ind.plot(R_km_arr[max_mass_idx], M_solar_arr[max_mass_idx],
                           'ko', markersize=7, label='Max mass')

        ax_mr_ind.set(
            xlabel='Radius [km]',
            ylabel=r'Mass [$M_\odot$]',
            title=f'M–R for {plot_base}'
        )
        ax_mr_ind.set_xlim(right=15)
        ax_mr_ind.grid(True, linestyle='--')
        ax_mr_ind.legend()
        fig_mr_ind.tight_layout()
        fig_mr_ind.savefig(os.path.join(Indiv_MR_Output, f'MR_{plot_base}.png'))
        plt.close(fig_mr_ind)

        # Λ–M plot
        fig_lm_ind, ax_lm_ind = plt.subplots(figsize=(8, 6))
        ax_lm_ind.plot(M_solar_arr, Lambda_arr, ':', color='grey', alpha=0.6, label='Full sequence')
        if np.any(stable_mask):
            ax_lm_ind.plot(M_solar_arr[stable_mask], Lambda_arr[stable_mask],
                           'C1.-', label='Stable')
        if np.any(unstable_mask):
            ax_lm_ind.plot(M_solar_arr[unstable_mask], Lambda_arr[unstable_mask],
                           'C0.--', label='Unstable')
        if max_mass_idx < len(M_solar_arr):
            ax_lm_ind.plot(M_solar_arr[max_mass_idx], Lambda_arr[max_mass_idx],
                           'ko', markersize=7, label='Max mass')

        ax_lm_ind.set(
            xlabel=r'Mass [$M_\odot$]',
            ylabel=r'Tidal deformability $\Lambda$',
            title=rf'$\Lambda(M)$ for {plot_base}',
            yscale='log'
        )
        ax_lm_ind.grid(True, which='both', linestyle='--')
        ax_lm_ind.legend()
        fig_lm_ind.tight_layout()
        fig_lm_ind.savefig(os.path.join(Indiv_LM_Output, f'LM_{plot_base}.png'))
        plt.close(fig_lm_ind)

        print("  Saved individual M–R and Λ–M plots.")

else:
    print("Skipping TOV step: EoS_Generations = 0 or no hybrid EoS generated.")

print("\nPart 2 complete")

# Part 3: Combined Mass-Radius and Tidal Deformability–Mass plots.


print("\n Part 3: Combined Mass-Radius and Tidal Deformability–Mass plots.")

def get_transition_mass(eos_path, eos_data, trans_dict):
    """Approximate mass where central pressure crosses that where the Phase Transition occurs."""
    key = os.path.basename(eos_path)
    props = trans_dict.get(key, {})
    p_trans_mev = props.get('p_trans', None)

    if p_trans_mev is None:
        return None

    star_data = eos_data['data']
    if star_data.size == 0:
        return None

    # Sort by central pressure and find first with p_c >= P_trans.
    idx = np.argsort(star_data[:, 3])
    sorted_data = star_data[idx]
    mask = sorted_data[:, 3] >= p_trans_mev

    if np.any(mask):
        return sorted_data[mask][0, 0]  # Mass of first star above transition pressure.
    return None

fig_mr_all, ax_mr_all = plt.subplots(figsize=(10, 8))
fig_lm_all, ax_lm_all = plt.subplots(figsize=(10, 8))

if not tov_results:
    print("No TOV results available; combined plots will be empty.")
else:
    num_eos = len(tov_results)
    colors = plt.cm.viridis(np.linspace(0, 1, num_eos))

    for i, (path, eos_dict) in enumerate(tov_results.items()):
        data = eos_dict['data']
        p_c_at_max_mass = eos_dict['p_c_at_max_mass']

        M_solar_arr, R_km_arr, Lambda_arr, p_c_mev_arr, _ = data.T

        unstable_mask = p_c_mev_arr > p_c_at_max_mass
        stable_mask   = ~unstable_mask
        color = colors[i]

        # Mass-Radius curves.
        ax_mr_all.plot(R_km_arr, M_solar_arr, ':', color=color, alpha=0.5)
        if np.any(stable_mask):
            ax_mr_all.plot(R_km_arr[stable_mask],   M_solar_arr[stable_mask],
                           '-', color=color, alpha=0.9)
        if np.any(unstable_mask):
            ax_mr_all.plot(R_km_arr[unstable_mask], M_solar_arr[unstable_mask],
                           '--', color=color, alpha=0.9)

        # Tidal Deformability curves.
        ax_lm_all.plot(M_solar_arr, Lambda_arr, ':', color=color, alpha=0.5)
        if np.any(stable_mask):
            ax_lm_all.plot(M_solar_arr[stable_mask],   Lambda_arr[stable_mask],
                           '-', color=color, alpha=0.9)
        if np.any(unstable_mask):
            ax_lm_all.plot(M_solar_arr[unstable_mask], Lambda_arr[unstable_mask],
                           '--', color=color, alpha=0.9)

        # Plots line where the first mass above Phase Transition.
        trans_mass = get_transition_mass(path, eos_dict, Phase_Trans_Prop)
        if trans_mass is not None:
            idx_closest = np.argmin(np.abs(M_solar_arr - trans_mass))
            ax_mr_all.axvline(x=R_km_arr[idx_closest], color=color, linestyle=':', alpha=0.4)
            ax_lm_all.axvline(x=trans_mass,            color=color, linestyle=':', alpha=0.4)

    style_handles = [
        Line2D([0], [0], color='gray', lw=1, linestyle='-',  label='Stable'),
        Line2D([0], [0], color='gray', lw=1, linestyle='--', label='Unstable'),
        Line2D([0], [0], color='gray', lw=1, linestyle=':',  label='Full sequence'),
        Line2D([0], [0], color='gray', lw=1, linestyle=':',  label='Transition marker'),
    ]

    ax_mr_all.set(
        xlabel='Radius [km]',
        ylabel=r'Mass [$M_\odot$]',
        title='Hybrid stars: combined M–R'
    )
    ax_mr_all.set_xlim(right=15)
    ax_mr_all.grid(True, linestyle='--')
    ax_mr_all.legend(handles=style_handles, loc='lower right', title='Branches')
    fig_mr_all.tight_layout()
    fig_mr_all.savefig('Mass_vs_Radius_FINAL.png')

    ax_lm_all.set(
        xlabel=r'Mass [$M_\odot$]',
        ylabel=r'Tidal deformability $\Lambda$',
        title='Hybrid stars: combined $\Lambda(M)$',
        yscale='log'
    )
    ax_lm_all.grid(True, which='both', linestyle='--')
    ax_lm_all.legend(handles=style_handles, loc='lower right', title='Branches')
    fig_lm_all.tight_layout()
    fig_lm_all.savefig('Lambda_vs_Mass_FINAL.png')

plt.close('all')

print("\nComplete.")
