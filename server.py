from flask import Flask, render_template, request, jsonify
import numpy as np
from scipy import integrate
from scipy.spatial import ConvexHull
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import base64
import io, os
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
    
    def simulate_trajectory_only(self, params):
        """Simplified simulation for footprint calculation - trajectory only"""
        # Extract parameters
        v0 = params['v0'] * 1000  # Convert km/s to m/s
        rolld0 = params['rolld0']
        gammad0 = params['gammad0']
        kappad0 = params['kappad0']
        r0 = params['r0'] * 1000  # Convert km to m
        cr0 = params['cr0'] * 1000  # Convert km to m
        t0 = params['t0']
        payload = params['payload']
        betaMetric = params['beta']
        LtoD = params['LtoD']
        
        # Unit conversions
        # betaMetric = beta * 47.9
        betaog = betaMetric / 9.81
        coeff = 1 / betaog
        gamma0 = (gammad0 * np.pi) / 180
        kappa0 = (kappad0 * np.pi) / 180
        roll0 = (rolld0 * np.pi) / 180
        
        # Initialize variables
        t = t0
        v = v0
        gamma = gamma0
        kappa = kappa0
        roll = roll0
        psi = r0 / self.Rearth
        omega = cr0 / self.Rearth
        pathlength = 0
        
        # Initial altitude calculation
        h0 = -6970 * np.log(((9.81 - ((v ** 2)/self.Rearth)) * betaog * 2) / (1.46 * (v ** 2) * LtoD))
        h = max(1000, h0)
        h = self.eq_alt(v, h, LtoD, coeff)
        
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
            'gamma_deg': [],
            'kappa_deg': [],
            'roll_switched': False  # Track if roll was switched
        }
        
        # Main simulation loop
        step_count = 0
        max_steps = int(tEND / deltat)
        tprint = dtprint
        roll_switched = False
        
        while t <= tEND and psi * self.Rearth <= maxrange and h > 0 and step_count < max_steps:
            step_count += 1
            
            rho = self.density(h)
            
            # Check if we should switch roll angle (when kappa reaches 90 degrees)
            kappa_deg = kappa * 180 / np.pi
            if not roll_switched and abs(kappa_deg) >= 90:
                roll = 0.0  # Switch to zero roll for maximum crossrange
                roll_switched = True
            
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
            
            # Integration using midpoint method
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
            
            # Calculate values at t + deltat
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
            
            # Calculate range and crossrange
            cosa = (np.sin(psi))**2 + np.cos(2*omega) * (np.cos(psi))**2
            cosa = np.clip(cosa, -1.0, 1.0)
            crange_earth = 0.5 * self.Rearth * np.arccos(cosa)
            
            sine = np.cos(omega) * np.cos(psi) / ((1+cosa)/2)**0.5
            sine = np.clip(sine, -1.0, 1.0)
            range_earth = (1.5708 - np.arcsin(sine)) * self.Rearth
            
            # Store results at print intervals
            if (t + (deltat / 2)) >= tprint:
                results['time'].append(t)
                results['altitude'].append(h / 1000)  # Convert to km
                results['velocity'].append(v / 1000)  # Convert to km/s
                results['range'].append(range_earth / 1000)  # Convert to km
                results['crossrange'].append(crange_earth / 1000)  # Convert to km
                results['gamma_deg'].append(gamma * 180 / np.pi)
                results['kappa_deg'].append(kappa * 180 / np.pi)
                
                tprint = tprint + dtprint
        
        results['roll_switched'] = roll_switched
        return results
    
    def calculate_footprint(self, params):
        """Calculate the footprint area by varying roll angle"""
        # Roll angles to test - include both positive and negative for full footprint
        positive_angles = np.linspace(0, 90, 10)  # 10 angles from 0 to 90
        negative_angles = np.linspace(-90, -5, 9)  # 9 angles from -90 to -5 (skip 0 to avoid duplicate)
        roll_angles = np.concatenate([negative_angles, positive_angles])
        
        print(f"Calculating footprint for roll angles: {roll_angles}")
        
        all_trajectories = []
        final_points = []
        
        for roll_angle in roll_angles:
            # Set the roll angle for this trajectory
            traj_params = params.copy()
            traj_params['rolld0'] = roll_angle
            
            print(f"Processing roll angle: {roll_angle}")
            
            # Run trajectory simulation
            try:
                trajectory = self.simulate_trajectory_only(traj_params)
                
                if trajectory['range'] and trajectory['crossrange']:
                    print(f"Trajectory for {roll_angle}° successful: {len(trajectory['range'])} points")
                    
                    all_trajectories.append({
                        'roll_angle': float(roll_angle),  # Ensure it's a Python float
                        'range': trajectory['range'],
                        'crossrange': trajectory['crossrange'],
                        'roll_switched': trajectory.get('roll_switched', False)
                    })
                    
                    # Store final point for footprint calculation
                    final_range = trajectory['range'][-1]
                    final_crossrange = trajectory['crossrange'][-1]
                    final_points.append([final_range, final_crossrange])
                    
                else:
                    print(f"Trajectory for {roll_angle}° failed: empty range or crossrange")
                    
            except Exception as e:
                print(f"Error with roll angle {roll_angle}: {e}")
                continue
        
        print(f"Successfully calculated {len(all_trajectories)} trajectories")
        
        if len(final_points) < 3:
            raise ValueError("Not enough valid trajectories to calculate footprint")
        
        # Add origin point
        final_points.append([0, 0])

        # Calculate footprint area using convex hull
        final_points = np.array(final_points)
        
        # Calculate convex hull
        try:
            hull = ConvexHull(final_points)
            footprint_area = hull.volume  # In 2D, volume gives area
            print(f"Calculated footprint area: {footprint_area} km²")
        except Exception as e:
            print(f"ConvexHull calculation failed: {e}")
            # Fallback: approximate area using bounding box
            min_range = np.min(final_points[:, 0])
            max_range = np.max(final_points[:, 0])
            min_crossrange = np.min(final_points[:, 1])
            max_crossrange = np.max(final_points[:, 1])
            footprint_area = (max_range - min_range) * (max_crossrange - min_crossrange) * 0.8
            hull = None
            print(f"Using fallback area calculation: {footprint_area} km²")
        
        return {
            'trajectories': all_trajectories,
            'final_points': final_points.tolist(),
            'footprint_area': footprint_area,
            'hull': hull.vertices.tolist() if hull else None
        }

    def simulate(self, params):
        """Main simulation function"""
        # Force initial conditions to zero
        params['r0'] = 0      # Initial range always 0
        params['cr0'] = 0     # Initial crossrange always 0  
        params['t0'] = 0      # Initial time always 0

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
        # emis = params['emis']
        # distance1 = params['distance1']
        # distance2 = params['distance2']
        # lam1 = params['lam1'] * 1e-6  # Convert um to m
        # lam2 = params['lam2'] * 1e-6
        # lam3 = params['lam3'] * 1e-6
        # lam4 = params['lam4'] * 1e-6

        roll_changes = params.get('rollChanges', [])
        roll_changes_dict = {}
        if roll_changes:
            for change in roll_changes:
                roll_changes_dict[change['time']] = change['angle'] * np.pi / 180  # Convert to radians
            print(f"Roll changes loaded: {roll_changes_dict}")
        
        # Vehicle geometry
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
        
        # Heating coefficients 
        # Csp = 0.000183 / (noserad ** 0.5)           # for stagnation point
        
        # Cphi1 = np.cos(phi1rad)
        # Sphi1 = np.sin(phi1rad)
        # Cphi2 = np.cos(phi2rad)
        # Sphi2 = np.sin(phi2rad)
        
        # # Laminar coefficients
        # Clam1 = 0.0000253 * (Cphi1 ** 0.5) * (Sphi1) / (distance1 ** 0.5)
        # Clam2 = 0.0000253 * (Cphi2 ** 0.5) * (Sphi2) / (distance2 ** 0.5)
        
        # # Turbulent coefficients
        # Cthv1 = 0.000022 * (Cphi1 ** 2.08) * (Sphi1 ** 1.6) / (distance1 ** 0.2)
        # Cthv2 = 0.000022 * (Cphi2 ** 2.08) * (Sphi2 ** 1.6) / (distance2 ** 0.2)
        # Ctlv1 = 0.000389 * (Cphi1 ** 1.78) * (Sphi1 ** 1.6) / (distance1 ** 0.2)
        # Ctlv2 = 0.000389 * (Cphi2 ** 1.78) * (Sphi2 ** 1.6) / (distance2 ** 0.2)

        
        # Initialize variables
        t = t0
        v = v0
        gamma = gamma0
        kappa = kappa0
        roll = roll0
        psi = r0 / self.Rearth
        omega = cr0 / self.Rearth
        pathlength = 0
        
        # Initial altitude calculation 
        h0 = -6970 * np.log(((9.81 - ((v ** 2)/self.Rearth)) * betaog * 2) / (1.46 * (v ** 2) * LtoD))
        h = max(1000, h0)  # Ensure reasonable starting altitude
        h = self.eq_alt(v, h, LtoD, coeff)
        
        # # Initial temperatures
        # Tsp = 3500
        # Tw1 = 2100
        # Tw2 = 1700
        
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
            'roll_deg': []
        }
        
        # Critical Reynolds number for transition
        Re_critical = 500000
        
        # Main simulation loop
        step_count = 0
        max_steps = int(tEND / deltat)
        tprint = dtprint
        
        while t <= tEND and psi * self.Rearth <= maxrange and h > 0 and step_count < max_steps:
            step_count += 1

            # Check for roll angle changes at current time
            if roll_changes_dict:
                # Find the most recent roll change that should be active
                active_roll = roll0  # Default to initial roll
                for change_time in sorted(roll_changes_dict.keys()):
                    if t >= change_time:
                        active_roll = roll_changes_dict[change_time]
                    else:
                        break
                
                # Only update if different from current roll
                if abs(roll - active_roll) > 1e-6:  # Small tolerance for floating point comparison
                    roll = active_roll
                    print(f"Roll angle changed to {roll * 180 / np.pi:.1f}° at time {t:.1f}s")
            
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
            
            # Integration using midpoint method 
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
            
            # Calculate range and crossrange 
            cosa = (np.sin(psi))**2 + np.cos(2*omega) * (np.cos(psi))**2
            cosa = np.clip(cosa, -1.0, 1.0)
            crange_earth = 0.5 * self.Rearth * np.arccos(cosa)
            
            sine = np.cos(omega) * np.cos(psi) / ((1+cosa)/2)**0.5
            sine = np.clip(sine, -1.0, 1.0)
            range_earth = (1.5708 - np.arcsin(sine)) * self.Rearth
            
            # Store results at print intervals
            if (t + (deltat / 2)) >= tprint:
                
                results['time'].append(t)
                results['altitude'].append(h / 1000)  # Convert to km
                results['velocity'].append(v / 1000)  # Convert to km/s
                results['range'].append(range_earth / 1000)  # Convert to km
                results['crossrange'].append(crange_earth / 1000)  # Convert to km
                results['pathlength'].append(pathlength / 1000)  # Convert to km
                results['gamma_deg'].append(gamma * 180 / np.pi)  # Convert to degrees
                results['kappa_deg'].append(kappa * 180 / np.pi)  # Convert to degrees
                results['roll_deg'] = getattr(results, 'roll_deg', [])  # Initialize if not exists
                results['roll_deg'].append(roll * 180 / np.pi)  # Convert to degrees

                
                tprint = tprint + dtprint
        
        return results

    # def reynolds_number(self, rho, v, L):
    #     """Calculate Reynolds number for boundary layer transition"""
    #     # Dynamic viscosity calculation using Sutherland's law
    #     # This is a simplified approach for hypersonic flow
    #     T_ref = 273.15  # Reference temperature (K)
    #     mu_ref = 1.716e-5  # Reference dynamic viscosity (Pa⋅s)
    #     S = 110.4  # Sutherland constant (K)
        
    #     # Approximate temperature from altitude (simplified)
    #     if rho > 0:
    #         T = max(200, 288.15 - 0.0065 * (6370000 - self.Rearth))  # Rough approximation
    #     else:
    #         T = 200
        
    #     # Sutherland's law for dynamic viscosity
    #     mu = mu_ref * ((T / T_ref) ** 1.5) * ((T_ref + S) / (T + S))
        
    #     # Reynolds number
    #     Re = (rho * v * L) / mu
    #     return Re
  

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/simulate', methods=['POST'])
def simulate():
    try:
        data = request.get_json()
        simulator = HypersonicGlideSimulator()
        
        # Run simulation
        results = simulator.simulate(data)
        
        # Validate results
        if not results.get('range') or len(results['range']) == 0:
            return jsonify({
                'success': False,
                'error': 'Simulation returned no data.'
            })
        
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

@app.route('/footprint', methods=['POST'])
def calculate_footprint():
    try:
        print("Footprint endpoint called")
        data = request.get_json()
        print(f"Received footprint data: {data}")
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'No data received'
            }), 400
        
        simulator = HypersonicGlideSimulator()
        
        print("Starting footprint calculation...")
        footprint_data = simulator.calculate_footprint(data)
        print(f"Footprint calculation completed. Area: {footprint_data['footprint_area']}")
        
        print("Generating footprint plot...")
        footprint_plot = generate_footprint_plot(footprint_data)
        print("Footprint plot generated successfully")
        
        # Build footprint curves for ALL calculated trajectories
        footprint_curves = {}
        
        if 'trajectories' in footprint_data:
            print("Processing trajectories for footprint curves...")
            for trajectory in footprint_data['trajectories']:
                roll_angle = trajectory.get('roll_angle')
                if roll_angle is not None and 'range' in trajectory and 'crossrange' in trajectory:
                    # Convert roll_angle to string key for JavaScript compatibility
                    roll_angle_key = str(float(roll_angle))  # Ensure consistent formatting
                    footprint_curves[roll_angle_key] = []
                    
                    range_data = trajectory['range']
                    crossrange_data = trajectory['crossrange']
                    
                    # Make sure both arrays have the same length
                    min_length = min(len(range_data), len(crossrange_data))
                    
                    for i in range(min_length):
                        footprint_curves[roll_angle_key].append({
                            'range': float(range_data[i]),
                            'crossrange': float(crossrange_data[i])
                        })
                    
                    print(f"Added {len(footprint_curves[roll_angle_key])} points for roll angle {roll_angle}")
        
        print(f"Final footprint_curves keys: {list(footprint_curves.keys())}")
        
        response_data = {
            'success': True,
            'footprint_area': footprint_data['footprint_area'],
            'plot': footprint_plot,
            'trajectories_count': len(footprint_data['trajectories']),
            'final_points': footprint_data['final_points'],
            'hull_vertices': footprint_data['hull'] if footprint_data['hull'] else [],
            'footprint_data': {  
                'curves': footprint_curves
            }
        }
        
        print(f"Sending response with {len(footprint_curves)} roll angle curves")
        return jsonify(response_data)
        
    except Exception as e:
        print(f"Error in footprint calculation: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

def generate_plots(results):
    """Generate all simulation plots"""
    plots = {}
    
    try:
        # Set up the plotting style
        plt.style.use('default')
        
        # Plot 1: Altitude vs Range
        plt.figure(figsize=(10, 6))
        plt.plot(results['range'], results['altitude'], 'b-', linewidth=2)
        plt.xlabel('Range (km)', fontsize=12)
        plt.ylabel('Altitude (km)', fontsize=12)
        plt.title('Altitude vs Range', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        plots['altitude_vs_range'] = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
        
        # Plot 2: Velocity vs Range
        plt.figure(figsize=(10, 6))
        plt.plot(results['range'], results['velocity'], 'r-', linewidth=2)
        plt.xlabel('Range (km)', fontsize=12)
        plt.ylabel('Velocity (km/s)', fontsize=12)
        plt.title('Velocity vs Range', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        plots['velocity_vs_range'] = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
        
        # Plot 3: Time vs Range
        plt.figure(figsize=(10, 6))
        plt.plot(results['range'], results['time'], 'g-', linewidth=2)
        plt.xlabel('Range (km)', fontsize=12)
        plt.ylabel('Time (s)', fontsize=12)
        plt.title('Flight Time vs Range', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        plots['time_vs_range'] = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
        
        # Plot 4: Crossrange vs Range
        plt.figure(figsize=(10, 6))
        plt.plot(results['range'], results['crossrange'], 'm-', linewidth=2)
        plt.xlabel('Range (km)', fontsize=12)
        plt.ylabel('Crossrange (km)', fontsize=12)
        plt.title('Cross Range vs Range', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        plots['crossrange_vs_range'] = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
        
        # # Plot 5: Temperature 1 vs Range
        # plt.figure(figsize=(10, 6))
        # plt.plot(results['range'], results['T1'], 'orange', linewidth=2)
        # plt.xlabel('Range (km)', fontsize=12)
        # plt.ylabel('Temperature (K)', fontsize=12)
        # plt.title('Temperature 1 vs Range', fontsize=14, fontweight='bold')
        # plt.grid(True, alpha=0.3)
        # plt.tight_layout()
        
        # img_buffer = io.BytesIO()
        # plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        # img_buffer.seek(0)
        # plots['T1_vs_range'] = base64.b64encode(img_buffer.getvalue()).decode()
        # plt.close()
        
        # # Plot 6: Temperature 2 vs Range
        # plt.figure(figsize=(10, 6))
        # plt.plot(results['range'], results['T2'], 'brown', linewidth=2)
        # plt.xlabel('Range (km)', fontsize=12)
        # plt.ylabel('Temperature (K)', fontsize=12)
        # plt.title('Temperature 2 vs Range', fontsize=14, fontweight='bold')
        # plt.grid(True, alpha=0.3)
        # plt.tight_layout()
        
        # img_buffer = io.BytesIO()
        # plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        # img_buffer.seek(0)
        # plots['T2_vs_range'] = base64.b64encode(img_buffer.getvalue()).decode()
        # plt.close()
        
        # # Plot 7: IR DSP vs Range
        # plt.figure(figsize=(10, 6))
        # plt.plot(results['range'], results['IR_DSP'], 'purple', linewidth=2)
        # plt.xlabel('Range (km)', fontsize=12)
        # plt.ylabel('IR Intensity (W/m²)', fontsize=12)
        # plt.title('DSP IR Intensity vs Range', fontsize=14, fontweight='bold')
        # plt.grid(True, alpha=0.3)
        # plt.tight_layout()
        
        # img_buffer = io.BytesIO()
        # plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        # img_buffer.seek(0)
        # plots['IR_DSP_vs_range'] = base64.b64encode(img_buffer.getvalue()).decode()
        # plt.close()
        
        # # Plot 8: IR SBIRS vs Range
        # plt.figure(figsize=(10, 6))
        # plt.plot(results['range'], results['IR_SBIRS'], 'cyan', linewidth=2)
        # plt.xlabel('Range (km)', fontsize=12)
        # plt.ylabel('IR Intensity (W/m²)', fontsize=12)
        # plt.title('SBIRS IR Intensity vs Range', fontsize=14, fontweight='bold')
        # plt.grid(True, alpha=0.3)
        # plt.tight_layout()
        
        # img_buffer = io.BytesIO()
        # plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        # img_buffer.seek(0)
        # plots['IR_SBIRS_vs_range'] = base64.b64encode(img_buffer.getvalue()).decode()
        # plt.close()
        
    except Exception as e:
        print(f"Error generating plots: {e}")
    
    return plots

def generate_footprint_plot(footprint_data):
    """Generate footprint visualization plot"""
    try:
        plt.figure(figsize=(12, 10))
        
        # Separate positive and negative roll angle trajectories
        positive_trajectories = []
        negative_trajectories = []
        
        for traj in footprint_data['trajectories']:
            if traj['roll_angle'] >= 0:
                positive_trajectories.append(traj)
            else:
                negative_trajectories.append(traj)
        
        # Plot positive roll angles (positive crossrange)
        colors_pos = plt.cm.viridis(np.linspace(0, 1, len(positive_trajectories)))
        for i, traj in enumerate(positive_trajectories):
            label = f"Roll +{traj['roll_angle']:.0f}°"
            plt.plot(traj['range'], traj['crossrange'], 
                    color=colors_pos[i], linewidth=1.5, alpha=0.7, label=label)
        
        # Plot negative roll angles (negative crossrange)
        colors_neg = plt.cm.plasma(np.linspace(0, 1, len(negative_trajectories)))
        for i, traj in enumerate(negative_trajectories):
            label = f"Roll {traj['roll_angle']:.0f}°"
            # For negative roll angles, plot crossrange as negative to show on opposite side
            crossrange_values = [-cr if traj['roll_angle'] < 0 else cr for cr in traj['crossrange']]
            plt.plot(traj['range'], crossrange_values, 
                    color=colors_neg[i], linewidth=1.5, alpha=0.7, label=label)
        
        # Create all points for hull calculation including both sides
        all_final_points = []
        for traj in footprint_data['trajectories']:
            final_range = traj['range'][-1]
            final_crossrange = traj['crossrange'][-1]
            
            if traj['roll_angle'] >= 0:
                # Positive roll angles on positive side
                all_final_points.append([final_range, final_crossrange])
            else:
                # Negative roll angles on negative side
                all_final_points.append([final_range, -final_crossrange])
        
        # Add origin
        all_final_points.append([0, 0])
        
        # Plot footprint boundary if convex hull was calculated
        if footprint_data['hull'] and len(all_final_points) > 2:
            # Recalculate hull with properly positioned points
            try:
                from scipy.spatial import ConvexHull
                hull = ConvexHull(np.array(all_final_points))
                hull_points = np.array(all_final_points)
                hull_indices = hull.vertices.tolist() + [hull.vertices[0]]  # Close the hull
                
                hull_x = [hull_points[i][0] for i in hull_indices]
                hull_y = [hull_points[i][1] for i in hull_indices]
                
                plt.plot(hull_x, hull_y, 'r-', linewidth=3, alpha=0.8, label='Footprint Boundary')
                plt.fill(hull_x, hull_y, 'red', alpha=0.1)
            except:
                print("Could not plot convex hull")
        
        # Mark final impact points on both sides
        final_points_array = np.array(all_final_points[:-1])  # Exclude origin
        plt.scatter(final_points_array[:, 0], final_points_array[:, 1], 
                   c='red', s=50, marker='o', alpha=0.8, 
                   label='Impact Points')
        
        plt.xlabel('Range (km)', fontsize=12)
        plt.ylabel('Crossrange (km)', fontsize=12)
        plt.title(f'Glider Footprint Analysis\nFootprint Area: {footprint_data["footprint_area"]:,.0f} km²', 
                 fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        
        # Add legend with limited entries to avoid clutter
        handles, labels = plt.gca().get_legend_handles_labels()
        # Show only every 3rd roll angle plus boundary and impact points
        legend_handles = []
        legend_labels = []
        
        for i, (handle, label) in enumerate(zip(handles, labels)):
            if 'Roll' in label and i % 3 == 0:
                legend_handles.append(handle)
                legend_labels.append(label)
            elif 'Footprint' in label or 'Impact' in label:
                legend_handles.append(handle)
                legend_labels.append(label)
        
        plt.legend(legend_handles, legend_labels, loc='upper right', 
                  bbox_to_anchor=(1.15, 1), fontsize=10)
        plt.tight_layout()
        
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        plot_data = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
        
        return plot_data
        
    except Exception as e:
        print(f"Error generating footprint plot: {e}")
        import traceback
        traceback.print_exc()  # This will help debug the exact error
        return None

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5051))
    app.run(host='0.0.0.0', port=port, debug=True)                                                                                     