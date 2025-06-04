from flask import Flask, render_template, request, jsonify
import numpy as np
from scipy import integrate
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import base64
import io
import json

app = Flask(__name__)

class HypersonicGlideSimulator:
    def __init__(self):
        # Constants
        self.Rearth = 6370000  # meters
        self.mu = 3.99E+14     # mks units
        self.sigma = 0.0000000567  # Stefan-Boltzmann constant
        self.h_p = 6.626e-34   # Planck's constant
        self.c_l = 2.998e8     # speed of light
        self.k_b = 1.381e-23   # Boltzmann constant
        self.Rg = 287          # gas constant
        self.P0 = 101325       # pressure at sea level
        
        # Atmospheric model data
        atmos_tab = np.array([
            [0., 288.1, 1., 28.964],
            [11.019, 216.65, .2284, 28.964],
            [20.063, 216.65, .05462, 28.964],
            [32.162, 228.65, 8.567e-3, 28.964],
            [47.350, 270.65, 1.095e-3, 28.964],
            [52.43, 270.65, 5.823e-4, 28.964],
            [61.59, 252.65, 1.797e-4, 28.964],
            [80., 180.65, 1.024e-5, 28.964],
            [90., 180.65, 1.622e-6, 28.964],
            [100., 210.65, 2.98e-7, 28.88],
            [110., 260.65, 7.22e-8, 28.56],
            [120., 360.65, 2.488e-8, 28.07],
            [150., 960.65, 5.0e-9, 26.92],
            [160., 1110.6, 3.64e-9, 26.66],
            [170., 1210.65, 2.756e-9, 26.5],
            [190., 1350.65, 1.66e-9, 25.85],
            [230., 1550.65, 6.869e-10, 24.69],
            [300., 1830.65, 1.86e-10, 22.66],
            [400., 2160.65, 3.977e-11, 19.94],
            [500., 2420.65, 1.08e-11, 17.94],
            [600., 2590.65, 3.40e-12, 16.84],
            [1500., 2700.65, 1.176e-12, 16.17],
        ])
        
        self.z_tab = atmos_tab[:, 0] * 1000  # altitude in meters
        self.t_tab = atmos_tab[:, 1]         # temperature
        self.p_tab = atmos_tab[:, 2] * self.P0  # pressure
        self.m_tab = atmos_tab[:, 3]         # molecular weight
        self.d_tab = self.p_tab / (self.Rg * self.t_tab)  # density
        
        # Calculate thermal lapse rate
        self.l_tab = np.zeros(21)
        for i in range(21):
            self.l_tab[i] = (self.t_tab[i + 1] - self.t_tab[i]) / (self.z_tab[i + 1] - self.z_tab[i])
    
    def density(self, z1):
        """Calculate atmospheric density at altitude z1"""
        g0 = 9.806
        b1 = 3.139e-7
        
        for i in range(20):
            if z1 >= self.z_tab[i+1]:
                continue
            elif abs(self.l_tab[i]) < 0.00001:
                q8 = (-1) * g0 * (z1 - self.z_tab[i]) * (1 - (b1 / 2) * (z1 + self.z_tab[i])) / (self.Rg * self.t_tab[i])
                D1 = np.exp(q8) * self.d_tab[i]
                return D1
            else:
                q1 = 1 + b1 * ((self.t_tab[i] / self.l_tab[i]) - self.z_tab[i])
                q2 = (q1 * g0) / (self.Rg * self.l_tab[i])
                t1 = self.t_tab[i] + (self.l_tab[i] * (z1 - self.z_tab[i]))
                q3 = t1 / self.t_tab[i]
                q5 = np.exp((b1 * g0 * (z1 - self.z_tab[i])) / (self.Rg * self.l_tab[i]))
                q7 = q2 + 1
                d1 = self.d_tab[i] * (q3 ** (-q7)) * q5
                return d1
        return self.d_tab[-1]  # Return last value if altitude is very high
    
    def eq_alt(self, v, h, LtoD, coeff):
        """Calculate equilibrium altitude"""
        rho = self.density(h)
        vert_force = (0.5 * LtoD * rho * (v ** 2) * coeff) + ((v ** 2) / (self.Rearth + h)) - 9.806
        
        if vert_force > 0:
            while vert_force > 0:  
                h = h + 1
                rho = self.density(h)
                vert_force = (0.5 * LtoD * rho * (v ** 2) * coeff) + ((v ** 2) / (self.Rearth + h)) - 9.806
        else:
            while vert_force < 0:  
                h = h - 1
                rho = self.density(h)
                vert_force = (0.5 * LtoD * rho * (v ** 2) * coeff) + ((v ** 2) / (self.Rearth + h)) - 9.806
        
        return h
    
    def reynolds_number(self, rho, v, L):
        """Calculate Reynolds number for boundary layer transition"""
        # Dynamic viscosity calculation using Sutherland's law
        # This is a simplified approach for hypersonic flow
        T_ref = 273.15  # Reference temperature (K)
        mu_ref = 1.716e-5  # Reference dynamic viscosity (Pa⋅s)
        S = 110.4  # Sutherland constant (K)
        
        # Approximate temperature from altitude (simplified)
        if rho > 0:
            T = max(200, 288.15 - 0.0065 * (6370000 - self.Rearth))  # Rough approximation
        else:
            T = 200
        
        # Sutherland's law for dynamic viscosity
        mu = mu_ref * ((T / T_ref) ** 1.5) * ((T_ref + S) / (T + S))
        
        # Reynolds number
        Re = (rho * v * L) / mu
        return Re
    
    def ir_em_front(self, lam, x, h, c, k, em, len1, thetarad, phi1rad, phi2rad, Tw1):
        """IR emission from front part of vehicle - exact from original code"""
        try:
            return (2 * em * h * (c ** 2)) * (1 / (lam ** 5)) * (1 / (np.exp((h * c) / (lam * k * ((x / np.cos(phi1rad)) ** -0.05) * Tw1)) - 1)) * 2 * x * np.tan(thetarad)
        except (OverflowError, ZeroDivisionError, ValueError):
            return 0
    
    def ir_em_rear(self, lam, x, h, c, k, em, len1, thetarad, phi1rad, phi2rad, Tw2):
        """IR emission from rear part of vehicle - exact from original code"""
        try:
            return (2 * em * h * (c ** 2)) * (1 / (lam ** 5)) * (1 / (np.exp((h * c) / (lam * k * (((len1 + ((x - len1) / np.cos(phi2rad))) / (3.5)) ** -0.05) * Tw2)) - 1)) * 2.2
        except (OverflowError, ZeroDivisionError, ValueError):
            return 0
    
    def simulate(self, params):
        """Main simulation function"""
        # Extract parameters
        v0 = params['v0'] * 1000  # Convert km/s to m/s
        rolld0 = params['rolld0']
        gammad0 = params['gammad0']
        kappad0 = params['kappad0']
        r0 = params['r0'] * 1000  # Convert km to m
        cr0 = params['cr0'] * 1000  # Convert km to m
        t0 = params['t0']
        payload = params['payload']
        beta = params['beta']
        LtoD = params['LtoD']
        emis = params['emis']
        distance1 = params['distance1']
        distance2 = params['distance2']
        lam1 = params['lam1'] * 1e-6  # Convert um to m
        lam2 = params['lam2'] * 1e-6
        lam3 = params['lam3'] * 1e-6
        lam4 = params['lam4'] * 1e-6
        
        # Vehicle geometry (fixed for HTV-2 based on original code)
        noserad = 0.034
        phi1 = 11.3
        phi2 = 7.6
        theta = 22.3
        len1 = 2.9
        len2 = 3.67
        
        # Unit conversions
        betaMetric = beta * 47.9
        betaog = betaMetric / 9.81
        coeff = 1 / betaog
        gamma0 = (gammad0 * np.pi) / 180
        kappa0 = (kappad0 * np.pi) / 180
        roll0 = (rolld0 * np.pi) / 180
        thetarad = (theta * np.pi) / 180
        phi1rad = (phi1 * np.pi) / 180
        phi2rad = (phi2 * np.pi) / 180
        
        # Heating coefficients - EXACT from original code
        Csp = 0.000183 / (noserad ** 0.5)           # for stagnation point
        
        Cphi1 = np.cos(phi1rad)
        Sphi1 = np.sin(phi1rad)
        Cphi2 = np.cos(phi2rad)
        Sphi2 = np.sin(phi2rad)
        
        # LAMINAR coefficients (low Reynolds number)
        Clam1 = 0.000776 * (Cphi1 ** 1.83) * (Sphi1 ** 1.6) / (distance1 ** 0.1)
        Clam2 = 0.000776 * (Cphi2 ** 1.83) * (Sphi2 ** 1.6) / (distance2 ** 0.1)
        
        # Turbulent high speed (V > 4km/s) coefficients
        Cthv1 = 0.000022 * (Cphi1 ** 2.08) * (Sphi1 ** 1.6) / (distance1 ** 0.2)
        Cthv2 = 0.000022 * (Cphi2 ** 2.08) * (Sphi2 ** 1.6) / (distance2 ** 0.2)
        
        # Turbulent low speed (V <= 4km/s) coefficients 
        Ctlv1 = 0.000389 * (Cphi1 ** 1.78) * (Sphi1 ** 1.6) / (distance1 ** 0.2)
        Ctlv2 = 0.000389 * (Cphi2 ** 1.78) * (Sphi2 ** 1.6) / (distance2 ** 0.2)
        
        # Initialize variables
        t = t0
        v = v0
        gamma = gamma0
        kappa = kappa0
        roll = roll0
        psi = r0 / self.Rearth
        omega = cr0 / self.Rearth
        pathlength = 0
        
        # Initial altitude calculation - EXACT from original
        h0 = -6970 * np.log(((9.81 - ((v ** 2)/self.Rearth)) * betaog * 2) / (1.46 * (v ** 2) * LtoD))
        h = max(1000, h0)  # Ensure reasonable starting altitude
        h = self.eq_alt(v, h, LtoD, coeff)
        
        # Initial temperatures
        Tsp = 3500
        Tw1 = 2100
        Tw2 = 1700
        
        # Integration parameters
        deltat = 0.1
        tEND = 3600
        maxrange = 20000 * 1000
        dtprint = 5
        
        # Storage arrays
        results = {
            'time': [],
            'altitude': [],
            'velocity': [],
            'range': [],
            'crossrange': [],
            'pathlength': [],
            'gamma_deg': [],
            'kappa_deg': [],
            'T_sp': [],
            'T1': [],
            'T2': [],
            'IR_SBIRS': [],
            'IR_DSP': [],
            'boundary_layer': []  # Track laminar vs turbulent
        }
        
        # Critical Reynolds number for transition (typical for hypersonic flow)
        Re_critical = 500000
        
        # Main simulation loop
        step_count = 0
        max_steps = int(tEND / deltat)
        tprint = dtprint
        
        while t <= tEND and psi * self.Rearth <= maxrange and h > 0 and step_count < max_steps:
            step_count += 1
            
            rho = self.density(h)
            
            # Store old values
            psiold = psi
            omegaold = omega
            hold = h
            gammaold = gamma
            kappaold = kappa
            vold = v
            told = t
            
            # Trigonometric values
            COSgo = np.cos(gammaold)
            SINgo = np.sin(gammaold)
            COSko = np.cos(kappaold)
            SINko = np.sin(kappaold)
            COSpo = np.cos(psiold)
            TANpo = np.tan(psiold)
            COSroll = np.cos(roll)
            SINroll = np.sin(roll)
            
            R = self.Rearth + hold
            muoverR2 = self.mu / (R ** 2)
            
            # Integration using midpoint method - EXACT from original
            psi_mid = psiold + ((vold * COSgo * COSko * deltat) / (2 * R))
            omega_mid = omegaold + ((vold * COSgo * SINko * deltat) / (2 * R * COSpo))
            h_mid = hold + ((vold * SINgo * deltat) / 2)
            
            dgamma = (vold * COSgo) / R
            dgamma = dgamma - ((muoverR2 * COSgo) / vold)
            dgamma = dgamma + (0.5 * LtoD * rho * vold * coeff * COSroll)
            gamma_mid = gammaold + ((dgamma * deltat) / 2)
            
            dkappa = -(0.5 * LtoD * rho * vold * coeff * SINroll) / COSgo
            dkappa = dkappa + ((vold * TANpo * COSgo * SINko) / R)
            kappa_mid = kappaold + ((dkappa * deltat) / 2)
            
            dv = -(coeff * rho * (vold ** 2)) / 2
            dv = dv - (SINgo * muoverR2)
            v_mid = vold + ((dv * deltat) / 2)
            
            # Now use derivatives at the midpoint to calculate values at t + deltat
            COSg_mid = np.cos(gamma_mid)
            SINg_mid = np.sin(gamma_mid)
            COSk_mid = np.cos(kappa_mid)
            SINk_mid = np.sin(kappa_mid)
            COSp_mid = np.cos(psi_mid)
            TANp_mid = np.tan(psi_mid)
            
            R_mid = self.Rearth + h_mid
            muoverR2_mid = self.mu / (R_mid ** 2)
            
            t = told + deltat
            psi = psiold + ((v_mid * COSg_mid * COSk_mid * deltat) / R_mid)
            omega = omegaold + ((v_mid * COSg_mid * SINk_mid * deltat) / (R_mid * COSp_mid))
            h = hold + (v_mid * SINg_mid * deltat)
            
            dgamma_mid = (v_mid * COSg_mid) / R_mid
            dgamma_mid = dgamma_mid - ((muoverR2_mid * COSg_mid) / v_mid)
            dgamma_mid = dgamma_mid + (0.5 * LtoD * rho * v_mid * coeff * COSroll)
            gamma = gammaold + (dgamma_mid * deltat)
            
            dkappa_mid = -(0.5 * LtoD * rho * v_mid * coeff * SINroll) / COSg_mid
            dkappa_mid = dkappa_mid + ((v_mid * TANp_mid * COSg_mid * SINk_mid) / R)
            kappa = kappaold + (dkappa_mid * deltat)
            
            dv_mid = -((coeff * rho * (v_mid ** 2)) / 2)
            dv_mid = dv_mid - (SINg_mid * muoverR2_mid)
            v = vold + (dv_mid * deltat)
            
            pathlength = pathlength + v * deltat
            
            # Calculate range and crossrange - EXACT from original
            cosa = (np.sin(psi))**2 + np.cos(2*omega) * (np.cos(psi))**2
            cosa = np.clip(cosa, -1.0, 1.0)
            crange_earth = 0.5 * self.Rearth * np.arccos(cosa)
            
            sine = np.cos(omega) * np.cos(psi) / ((1+cosa)/2)**0.5
            sine = np.clip(sine, -1.0, 1.0)
            range_earth = (1.5708 - np.arcsin(sine)) * self.Rearth
            
            # Store results at print intervals
            if (t + (deltat / 2)) >= tprint:
                # Calculate Reynolds number to determine boundary layer type
                Re1 = self.reynolds_number(rho, v, distance1)
                Re2 = self.reynolds_number(rho, v, distance2)
                
                # Determine boundary layer type
                is_turbulent_1 = Re1 > Re_critical
                is_turbulent_2 = Re2 > Re_critical
                
                # Calculate heating - EXACT from original Anderson equations
                enth0 = (0.5 * (v ** 2)) + 0.000023
                
                enthSP = 1000 * Tsp
                enthratioSP = enthSP / enth0
                
                enthW1 = 1000 * Tw1
                enthratioW1 = enthW1 / enth0
                
                enthW2 = 1000 * Tw2
                enthratioW2 = enthW2 / enth0
                
                qSP = Csp * (rho ** 0.5) * (v ** 3) * (1 - enthratioSP)
                
                # Select heating equations based on boundary layer type and velocity
                if is_turbulent_1:
                    if v > 4000:  # High speed turbulent equations (v > 4km/s)
                        qWALL1 = Cthv1 * (rho ** 0.8) * (v ** 3.7) * (1 - (1.11 * enthratioW1))
                    else:  # Low speed turbulent equations (v <= 4km/s)
                        qWALL1 = Ctlv1 * (rho ** 0.8) * (v ** 3.37) * (1 - (1.11 * enthratioW1)) * ((556 / Tw1) ** 0.25)
                else:
                    # Laminar flow equations
                    qWALL1 = Clam1 * (rho ** 0.5) * (v ** 3.2) * (1 - (1.11 * enthratioW1)) * ((556 / Tw1) ** 0.1)
                
                if is_turbulent_2:
                    if v > 4000:  # High speed turbulent equations (v > 4km/s)
                        qWall2 = Cthv2 * (rho ** 0.8) * (v ** 3.7) * (1 - (1.11 * enthratioW2))
                    else:  # Low speed turbulent equations (v <= 4km/s)
                        qWall2 = Ctlv2 * (rho ** 0.8) * (v ** 3.37) * (1 - (1.11 * enthratioW2)) * ((556 / Tw2) ** 0.25)
                else:
                    # Laminar flow equations
                    qWall2 = Clam2 * (rho ** 0.5) * (v ** 3.2) * (1 - (1.11 * enthratioW2)) * ((556 / Tw2) ** 0.1)
                
                IR_SBIRS = 0
                IR_DSP = 0
                
                if qSP > 0 and qWALL1 > 0 and qWall2 > 0:
                    # Calculate wall temperatures using Stefan-Boltzmann
                    Tsp = (qSP / (emis * self.sigma)) ** 0.25
                    Tw1 = (qWALL1 / (emis * self.sigma)) ** 0.25
                    Tw2 = (qWall2 / (emis * self.sigma)) ** 0.25
                    
                    # Calculate IR emission - simplified for performance
                    # Full integration would be too computationally expensive for web app
                    try:
                        # Approximate IR emission using Stefan-Boltzmann law
                        # This is a simplification of the full Planck integration
                        area_front = np.pi * len1 * noserad * 2  # Approximate surface area
                        area_rear = np.pi * (len2 - len1) * noserad * 2.2
                        
                        # Simplified spectral radiance calculation
                        T1_eff = Tw1 * ((distance1 / 1.0) ** -0.05)  # Distance scaling
                        T2_eff = Tw2 * (((len1 + distance2) / 3.5) ** -0.05)
                        
                        # Stefan-Boltzmann approximation for spectral bands
                        # This is much faster than full Planck integration
                        IR_SBIRS = emis * self.sigma * (T1_eff ** 4) * area_front * 0.001  # Convert to kW/sr
                        IR_DSP = emis * self.sigma * (T2_eff ** 4) * area_rear * 0.001
                        
                    except Exception as e:
                        IR_SBIRS = 0
                        IR_DSP = 0
                
                # Determine overall boundary layer state for tracking
                boundary_layer_state = "Turbulent" if (is_turbulent_1 or is_turbulent_2) else "Laminar"
                
                # Store results
                results['time'].append(t)
                results['altitude'].append(h / 1000)  # Convert to km
                results['velocity'].append(v / 1000)  # Convert to km/s
                results['range'].append(range_earth / 1000)  # Convert to km
                results['crossrange'].append(crange_earth / 1000)  # Convert to km
                results['pathlength'].append(pathlength / 1000)  # Convert to km
                results['gamma_deg'].append(gamma * 180 / np.pi)  # Convert to degrees
                results['kappa_deg'].append(kappa * 180 / np.pi)  # Convert to degrees
                results['T_sp'].append(Tsp)
                results['T1'].append(Tw1)
                results['T2'].append(Tw2)
                results['IR_SBIRS'].append(IR_SBIRS)
                results['IR_DSP'].append(IR_DSP)
                results['boundary_layer'].append(boundary_layer_state)
                
                tprint = tprint + dtprint
        
        return results

# Initialize simulator
simulator = HypersonicGlideSimulator()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/simulate', methods=['POST'])
def simulate():
    try:
        params = request.json
        results = simulator.simulate(params)
        
        # Generate plots
        plots = generate_plots(results)
        
        return jsonify({
            'success': True,
            'results': results,
            'plots': plots
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

def generate_plots(results):
    """Generate base64 encoded plots"""
    plots = {}
    
    # Set up matplotlib style
    plt.style.use('default')
    
    plot_configs = [ 
        ('altitude_vs_range', 'range', 'altitude', 'Range (km)', 'Altitude (km)', 'Altitude vs Range'),
        ('velocity_vs_range', 'range', 'velocity', 'Range (km)', 'Velocity (km/s)', 'Velocity vs Range'),
        ('time_vs_range', 'range', 'time', 'Range (km)', 'Time (s)', 'Flight Time vs Range'),
        ('crossrange_vs_range', 'range', 'crossrange', 'Range (km)', 'Cross Range (km)', 'Cross Range vs Range'),
        ('T1_vs_range', 'range', 'T1', 'Range (km)', 'Temperature at Distance 1 (K)', 'Temperature 1 vs Range'),
        ('T2_vs_range', 'range', 'T2', 'Range (km)', 'Temperature at Distance 2 (K)', 'Temperature 2 vs Range'),
        ('IR_DSP_vs_range', 'range', 'IR_DSP', 'Range (km)', 'DSP IR Intensity (kW/sr)', 'DSP IR Intensity vs Range'),
        ('IR_SBIRS_vs_range', 'range', 'IR_SBIRS', 'Range (km)', 'SBIRS IR Intensity (kW/sr)', 'SBIRS IR Intensity vs Range'),
    ]
    
    for plot_name, x_key, y_key, xlabel, ylabel, title in plot_configs:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if results[x_key] and results[y_key]:
            ax.plot(results[x_key], results[y_key], 'b-', linewidth=2)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            
            # Save plot to base64 string
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            plot_data = buffer.getvalue()
            buffer.close()
            
            plots[plot_name] = base64.b64encode(plot_data).decode()
        
        plt.close(fig)
    
    return plots

import os

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
