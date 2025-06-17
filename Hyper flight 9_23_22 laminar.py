# HYP-HT-2 conversion from BASIC to Python 4/20

# A modifcation of HYPER-1 used to calculate glide curves for the HTV-2 
# vehicle, with heating and IR emission equations included

# A modification of BALLISTC that flies a vehicle at an altitude determined by 
# lift and drag.

# BALLISTC calculates ballistic trajectory from initial parameters (burnout 
# speed, angle from horizontal, range from launch point, height above earth, 
# and time since launch) in the presence of an atmosphere.

# Coordinates: Prints results in x and z, with origin at the center of the 
# earth, and also range along the earth and altitude above the earth.


# This version uses the full equations with lat and long, and then calculates
# range and crossrange from a point's lat/long using my expressions from 
# spehrical geometry


##### IN THIS VERSION PSI IS LONGITUDE, OMEGA IS LATITUDE, AND KAPPA IS
##### MEASURED FROM NORTH.






# **************************************************************************


import numpy as np
from scipy import integrate


# **************************************************************************

test = 0
# Set initial flight parameters:

v0 = 5.5 * 1000   # speed (in m/s)
rolld0 = 0      # vehicle orientation, angle from vertical (in degrees)

gammad0 = 0     # velocity direction, angle from horizontal (in degrees)
kappad0 = 0     # velocity direction, angle from forward (in degrees)
r0 = 0 * 1000     # range from launch point (in m)
cr0 = 0 * 500    # crossrange from launch point (in m)
t0 = 0            # time since launch (in s)
# h0 = 50 * 1000    # height above earth (in m)

maxrange = 20000 * 1000    # in m; tells program when to stop

# Set vehicle specifications:

payload = 1000      # vehicle mass in kg
beta = 2663         # in lb./ft^2;  ballistic parameter of RV
LtoD = 2.6          # lift to drag ratio
emis = 0.85         # emissivity of shell of vehicle

noserad = 0.034      # radius of vehicle nose (in m)
phi1 = 11.3         # top vertical angle of front 80% of HTV-2 (in degrees)
phi2 = 7.6          # top vertical angle of rear 20% of HTV-2 (in degrees)
theta = 22.3        # symmetric horizontal angle of nose

len1 = 2.9          # length of front 80% of HTV-2
len2 = 3.67         # length of front 80% and rear 20% of HTV-2

distance1 = 1       # distance from nose where heating is calc (in m)
distance2 = 3.5     # distance from nose where heating is calc (in m)


# **************************************************************************


# Define coefficients for heating equations from Anderson: these are heat 
# per time per area (note that coeffs in missile programs included area 
# of RV, making them heat per time)

Csp = 0.000183 / (noserad ** 0.5)           # for stagnation point

Cphi1 = np.cos(phi1 * (3.14159 / 180))
Sphi1 = np.sin(phi1 * (3.14159 / 180))

Cphi2 = np.cos(phi2 * (3.14159 / 180))
Sphi2 = np.sin(phi2 * (3.14159 / 180))

# for laminar
Cthv1 = 0.0000253 * (Cphi1 ** 0.5) * (Sphi1) / (distance1 ** 0.5)     
Cthv2 = 0.0000253 * (Cphi2 ** 0.5) * (Sphi2) / (distance2 ** 0.5)

# for turbulent low speed (V <= 4km/s)
#Ctlv1 = 0.000389 * (Cphi1 ** 1.78) * (Sphi1 ** 1.6) / (distance1 ** 0.2)
#Ctlv2 = 0.000389 * (Cphi2 ** 1.78) * (Sphi2 ** 1.6) / (distance2 ** 0.2)


# **************************************************************************


# define lower (1, 3) and upper (2, 4) wavelengths of IR bands of interest [m]
# SWIR is 1.4e-6 to 3e-6; MWIR is 3e-6 to 8e-6 m

lam1 = 1.4e-6     # [m]
lam2 = 3e-6
lam3 = 2.69e-6
lam4 = 2.95e-6


# **************************************************************************


# Perform simple unit conversions

betaMetric = beta * 47.9     # beta in N/m^2
betaog = betaMetric / 9.81   # beta in N/m^2 divided by g (accel of 
                             # gravity at h = 0)
coeff = 1 / betaog     # this is how drag coeff enters the eqs during reentry
gamma0 = (gammad0 * 3.14159) / 180     # convert gammad0 to radians
kappa0 = (kappad0 * 3.14159) / 180     # convert kappad0 to radians
roll0 = (rolld0 * 3.14159) / 180       # convert roll angle to radians
thetarad = (theta * (3.14159 / 180))
phi1rad = (phi1 * (3.14159 / 180))
phi2rad = (phi2 * (3.14159 / 180))

# Set misc constants

Rearth = 6370000       # in meters
mu = 3.99E+14          # in mks units; mu = G Me = g * Rearth^2
sigma = .0000000567    # Stefan-Boltzmann const in W/(m2K4)

h_p = 6.626e-34        # Planck's constant [J/s]
c_l = 2.998e8          # speed of light [m/s]
k_b = 1.381e-23        # Boltzmann constant [J/K]
em = 0.85              # emissivity of carbon, approximate

# Set integration timing parameters

tEND = 3600      # time in seconds at which program stops
dtprint = 5       # time interval in seconds between printing outputs

deltatend = 0.1        # time increment used in integration
deltatinit = 0.1       # time increment used for t < tinit sec
tinit = 1


# **************************************************************************


# print text file listing key parameters

prm = open('HTV2_flight-Lat-Long-R&XR.prm', 'w')

prm.write('HYP-HT-1 \n')
prm.write(f'initial gamma = {gammad0} degrees ({gamma0} radians) \n')
prm.write(f'deltat = {deltatinit} for 0 < t < {tinit} ; deltat = '
          f'{deltatend} for t > {tinit} \n')
prm.write(f'dtprint = {dtprint} \n \n')

prm.write('VEHICLE PARAMETERS: \n')
prm.write(f'beta = {beta} lb/ft^2 (note English units) \n')
prm.write(f'vehicle mass = {payload} kg \n')
prm.write(f'nose radius = {noserad} m \n')
prm.write(f'phi1 = {phi1} degrees ; phi2 = {phi2} degrees \n \n')

prm.write('HEATING CALC PARAMETERS: \n')
prm.write(f'distance1 = {distance1} m ; distance2 = {distance2} m \n')
prm.write(f'Csp = {Csp:.6e} ; Cthv1 = {Cthv1:.6e} ; Cthv2 = {Cthv2:.6e} \n')

prm.close()

dat = open('HTV2_flight-Lat-Long-R&XR.dat', 'w')

dat.write('x (km)        y (km)        z (km)        t (s)         rng (km)      c_rng (km)    alt (km)   pathlength'     
          '     v (km/s)      v_ang (deg)   h_ang (deg)   T_sp (K)      T1m (K)       T3.5m (K)'
          '     I_SB (kW/sr)   I_DS (kW/sr) \n \n')


# **************************************************************************


# initialize variables
    
t = t0
deltat = deltatinit
flagdeltat = 1         # used in changing deltat from deltatinit to deltatend
M = payload
v = v0
gamma = gamma0
kappa = kappa0
roll = roll0
psi = r0 / Rearth      # range angle: range = psi * Rearth
omega = cr0 / Rearth   # crossrange angle: crossrange = omega * Rearth
range_earth = 0
pathlength = 0         #v integrated over t

h0 = -6970 * np.log(((9.81 - ((v ** 2)/Rearth)) * betaog * 2) / ( 1.46 * (v ** 2) * LtoD))
h = h0           # height above ground in meters, later refined with density model

Tsp = 3500       # in K; estimates of initial wall temp for heating calcs
Tw1 = 2100
Tw2 = 1700

tprint = dtprint   # tprint = time at which printing of output will next occur
gammap5 = gamma    # gamma = angle of the missile or RV with the local horizon




# **************************************************************************

# define functions: atmospheric density as a function of altitude, equilibrium
# flight altitude as a function of velocity, and radiant intensity as a
# function of surface temperatures

# define atmospheric density function
    
    # simple atmospheric model from Frank J. Regan "Re-entry Vehicle Dynamics"
    # pg.18 (MIT Aero Lib.TL1060.R43 1984). Calculates pressure, density, 
    # temperature and molecular weight for altitudes up to 700 km. Parameters 
    # are gas constant Rg (J/kg K), and the sea level values of: grav. const g0 
    # (m/s2), atmos pressure p0 (n/m2), molecular weight m0, and density do 
    # (kg/m3). First order grav constant is b (1/m). The subscripted 
    # variables are altitude z(i) (km), molecular temp t(i) (K), pressure p(i), 
    # molecular weight m(i) (kg/mole), density d(i), and thermal lapse rate 
    # L(i) (K/km). t9 is the temperature. Program was checked against 
    # original equations and measured values. In converting David's code, I
    # changed working variables t to t1, etc, to avoid confusion between 
    # working variables and tabulated arrays, which are identified by "_tab"

# tabulated data from book source
    
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
Rg = 287          # gas constant
P0 = 101325       # pressure at sea level

z_tab = atmos_tab[:,0] * 1000      # altitude, converted to meters
t_tab = atmos_tab[:,1]             # temperature
p_tab = atmos_tab[:,2] * P0        # pressure
m_tab = atmos_tab[:,3]             # molecular weight
d_tab = p_tab / (Rg * t_tab)                # density

    # set L, thermal lapse rate, which requires a loop because L_tab[i] is 
    # derived from a[i+1], etc.

l_tab = np.zeros(21)

for i in range(21):
    l_tab[i] = (t_tab[i + 1] - t_tab[i]) / (z_tab[i + 1] - z_tab[i])

# define function

def density(z1, z_tab=z_tab, t_tab=t_tab, p_tab=p_tab, m_tab=m_tab, d_tab=d_tab, l_tab=l_tab):

    Rg = 287
    g0 = 9.806
    # m0 = 28.964
    # P0 = 101325
    # d0 = 1.225
    b1 = 3.139e-7
    
    for i in range(20):
        if z1 >= z_tab[i+1]:     # cycle through z(i) until the one just below h
            continue
        elif abs(l_tab[i]) < 0.00001:       # test if dL/dh is 0
            t1 = t_tab[i]
            q8 = (-1) * g0 * (z1 - z_tab[i]) * (1 - (b1 / 2) * (z1 + z_tab[i])) / (Rg * t_tab[i])
            # p1 = np.exp(q8) * p_tab[i]
            D1 = np.exp(q8) * d_tab[i]
            # simple interpolation of molecular weight
            # m1 = m_tab[i] + (((m_tab[i+1] - m_tab[i]) / (z_tab[i + 1] - z_tab[i])) * (z1 - z_tab[i]))
            # t9 = (m1 * t1) / m0
            rho = D1  
            break
        else:
            q1 = 1 + b1 * ((t_tab[i] / l_tab[i]) - z_tab[i])
            q2 = (q1 * g0) / (Rg * l_tab[i])
            t1 = t_tab[i] + (l_tab[i] * (z1 - z_tab[i]))
            q3 = t1 / t_tab[i]
            # q4 = q3 ** (-q2)
            q5 = np.exp((b1 * g0 * (z1 - z_tab[i])) / (Rg * l_tab[i]))
            # q6 = q4 * q5
            # p1 = p_tab[i] * q6
            q7 = q2 + 1
            d1 = d_tab[i] * (q3 ** (-q7)) * q5
            # simple interpolation of molecular weight
            # m1 = m_tab[i] + (((m_tab[i+1] - m_tab[i]) / (z_tab[i + 1] - z_tab[i])) * (z1 - z_tab[i]))
            # t9 = (m1 * t1) / m0
            rho = d1
            break
        
    return rho


# **************************************************************************
        

def eq_alt(v=v, h=h, LtoD=LtoD, coeff=coeff, Rearth=Rearth):
    
    rho = density(h)
    
    # calculate the acceleration in the vertical direction for the given altitude
    # lift + centrifugal - gravity
    vert_force = (0.5 * LtoD * rho * (v ** 2) * coeff) + ((v ** 2) / (Rearth + h)) - 9.806 
    
    if vert_force > 0:
        while vert_force > 0:
            h = h + 1
            rho = density(h)
            vert_force = (0.5 * LtoD * rho * (v ** 2) * coeff) + ((v ** 2) / (Rearth + h)) - 9.806
    else:
        while vert_force < 0:
            h = h - 1
            rho = density(h)
            vert_force = (0.5 * LtoD * rho * (v ** 2) * coeff) + ((v ** 2) / (Rearth + h)) - 9.806
        
    return h


# **************************************************************************


# Define radiant intensity functions

def ir_em_front(lam, x, h, c, k, em, len1, thetarad, phi1rad, phi2rad, Tw1):
    # Equation returning radiant intensity [W/sr] of the front part of the vehicle
    # Planck's Law with temperature values from Anderson eq using Tw1 (at 1 m along vehicle body)
    return (2 * em * h * (c ** 2)) * (1 / (lam ** 5)) * (1 / (np.exp((h * c) / (lam * k * ((x / np.cos(phi1rad)) ** -0.05) * Tw1)) - 1)) * 2 * x * np.tan(thetarad)

def ir_em_rear(lam, x, h, c, k, em, len1, thetarad, phi1rad, phi2rad, Tw2):
    # Equation returning radiant intensity [W/sr] of the front part of the vehicle
    # Planck's Law with temperature values from Anderson eq using Tw2 (at 3.5 m along vehicle body)
    return (2 * em * h * (c ** 2)) * (1 / (lam ** 5)) * (1 / (np.exp((h * c) / (lam * k * (((len1 + ((x - len1) / np.cos(phi2rad))) / (3.5)) ** -0.05) * Tw2)) - 1)) * 2.2


# **************************************************************************


# set starting altitude to more accurate equilibrium value using density model
h = eq_alt()

#h = 50000

# *****************START OF BIG LOOP OVER TIME*******************************


while  t <= tEND and range_earth <= maxrange and h > 0:
    if (t + deltat / 2) >= tinit and flagdeltat == 1:  
        # using deltat/2 allows for rounding errors. This statement allows for 
        # smaller delta t near the end of flight
        deltat = deltatend
        flagdeltat = 0
    
    rho = density(h)      # calls density() function which passes atmos density rho
    
    psiold = psi
    omegaold = omega
    hold = h
    gammaold = gamma
    kappaold = kappa
    vold = v
    mold = M
    told = t
    
    # condition for initiating a mid flight turn
    # if [condition]:
    #     roll = 
    
    COSgo = np.cos(gammaold)     # for great circle flight case gamma = 0, otherwise cos(gammaold)
    SINgo = np.sin(gammaold)     # for great circle flight case gamma = 0, otherwise sin(gammaold)
    COSko = np.cos(kappaold)     # for straight flight case gamma = 0, otherwise cos(kappaold)
    SINko = np.sin(kappaold)     # for straight flight case gamma = 0, otherwise sin(kappaold)
   
    COSpo = np.cos(psiold)
    TANpo = np.tan(psiold)
    
    COSroll = np.cos(roll)
    SINroll = np.sin(roll)
    
    # ETAo = eta(hold, told)   # eta is the angle that sets lateral thrust angle during 
                               # boost; not used for hypersonic glide
    # Toverm = 0               # thrust over mass. For glide vehicle, thrust is 0
    R = Rearth + hold          # distance from center of earth
    muoverR2 = mu / (R ** 2)
    
    # Integration is a variant of Runge-Kutta method for better convergence (midpoint method).
    # First calculate values of parameters at midpoint:  t = told + deltat/2
    
    # t_mid = told + (deltat / 2)
    psi_mid = psiold + ((vold * COSgo * COSko * deltat) / (2 * R))
    omega_mid = omegaold + ((vold * COSgo * SINko * deltat) / (2 * R * COSpo))
    h_mid = hold + ((vold * SINgo * deltat) / 2)
    
    dgamma = (vold * COSgo) / R
    # dgamma = dgamma + ((Toverm * np.sin(ETAo)) / vold)   # no thrust for hypersonic glide case
    dgamma = dgamma - ((muoverR2 * COSgo) / vold)
    dgamma = dgamma + (0.5 * LtoD * rho * vold * coeff * COSroll)
    gamma_mid = gammaold + ((dgamma * deltat) / 2)
    
    dkappa = -(0.5 * LtoD * rho * vold * coeff * SINroll) / COSgo
    dkappa = dkappa + ((vold * TANpo * COSgo * SINko) / R) 
    kappa_mid = kappaold + ((dkappa * deltat) / 2)
    
    dv = 0    # = COS(ETAo) * Toverm       # no eta for hypersonic glide case
    dv = dv - (coeff * rho * (vold ** 2)) / 2
    dv = dv - (SINgo * muoverR2)
    v_mid = vold + ((dv * deltat) / 2)
    
    # Now use derivatives at the midpoint to calculate values at t + deltat
    
    COSg_mid = np.cos(gamma_mid)      # for hypersonic case, keep gamma = 0
    SINg_mid = np.sin(gamma_mid)      # for hypersonic case, keep gamma = 0
    COSk_mid = np.cos(kappa_mid)     
    SINk_mid = np.sin(kappa_mid)
    
    COSp_mid = np.cos(psi_mid)
    TANp_mid = np.tan(psi_mid)
    
    # ETA_mid = eta(h_mid, t_mid)     # no eta for hypersonic glide case
    R_mid = Rearth + h_mid
    # Toverm_mid = 0                  # no thrust for hypersonic glide case
    muoverR2_mid = mu / (R_mid ** 2)
    
    t = told + deltat
    psi = psiold + ((v_mid * COSg_mid * COSk_mid * deltat) / R_mid)
    omega = omegaold + ((v_mid * COSg_mid * SINk_mid * deltat) / (R_mid * COSp_mid))
    h = hold + (v_mid * SINg_mid * deltat)
    
    dgamma_mid = (v_mid * COSg_mid) / R_mid
    # dgamma_mid = dgamma_mid + ((Toverm_mid * np.sin(ETA_mid)) / v_mid)   # no thrust
    dgamma_mid = dgamma_mid - ((muoverR2_mid * COSg_mid) / v_mid)
    dgamma_mid = dgamma_mid + (0.5 * LtoD * rho * v_mid * coeff * COSroll)
    gamma = gammaold + (dgamma_mid * deltat)
    
    dkappa_mid = -(0.5 * LtoD * rho * v_mid * coeff * SINroll) / COSg_mid
    dkappa_mid = dkappa_mid + ((v_mid * TANp_mid * COSg_mid * SINk_mid) / R)
    kappa = kappaold + (dkappa_mid * deltat)
    
    dv_mid = 0    # = np.cos(ETA_mid) * Toverm_mid   # no thrust for hypersonic glide
    dv_mid = dv_mid - ((coeff * rho * (v_mid ** 2)) / 2)  # call rho at midpoint?
    dv_mid = dv_mid - (SINg_mid * muoverR2_mid)
    v = vold + (dv_mid * deltat)
    
    pathlength=pathlength + v*deltat

    
    # prevent turning backwards to maximize range and cross range
    
    #if psi*Rearth > 9500000:
        #roll = -45
    
    # if kappa_mid > 1.570796 and roll != 0:
    #     roll = 0
    #     kappa = 1.570796
        
    # if kappa_mid < -1.570796 and roll != 0:
    #     roll = 0
    #     kappa = -1.570796
    
    # if (1.5708-np.arcsin(np.cos(omega) * np.cos(psi) /((1+(((np.sin(psi))**2 + np.cos(2*omega) * (np.cos(psi))**2)))/2)**.5)) > 0.5 and test == 0:
    #     roll = 0
    #     kappa = 0
    #     test = 1
    
    
    # Set glide altitude of hypersonic vehicle for current value of v:
    # h = -6970 * np.log(((9.81 - ((v**2)/Rearth)) * betaog * 2) / ( 1.46 * (v ** 2) * LtoD))
    
    
    # *****************************************
    
    
    # for tprint interval: calculate position, range, body heat, IR emission, and 
    # print data points
    
    if (t + (deltat / 2)) >= tprint:     #(deltat/2 allows for rounding errors)
        
        #print()
        
        x = (Rearth + h) * np.sin(psi)
        y = (Rearth + h) * np.sin(omega)
        z = ((Rearth + h) * np.cos(psi)) - Rearth
        
        # OLD EXPRESSIONS
        # range_earth = Rearth * psi
        # crange_earth = Rearth * omega * np.cos(psi)

        cosa = (np.sin(psi))**2 + np.cos(2*omega) * (np.cos(psi))**2
        cosa = np.clip(cosa, -1.0, 1.0) #Added this to clamp cosa to valid range [-1, 1] to avoid arccos domain errors
        crange_earth = 0.5*Rearth*np.arccos(cosa)
        
        sine = np.cos(omega) * np.cos(psi) /((1+cosa)/2)**.5
        range_earth = (1.5708-np.arcsin(sine))*Rearth

        gammad = gamma * (180 / 3.14159)
        kappad = kappa * (180 / 3.14159)
        
        # Calculate heating (from Anderson, "Hypersonic and High T Gas Dynamics", 
        # p. 291)
        
        enth0 = (0.5 * (v ** 2)) + 0.000023         # enthalpy in J/kg
              
        enthSP = 1000 * Tsp                         # for stagnation point
        enthratioSP = enthSP / enth0
        
        enthW1 = 1000 * Tw1               # for first point on wall (at distance1)
        enthratioW1 = enthW1 / enth0
        
        enthW2 = 1000 * Tw2              # for second point on wall (at distance2)
        enthratioW2 = enthW2 / enth0
        
        qSP = Csp * (rho ** (0.5)) * (v ** 3) * (1 - enthratioSP)   # stagnation pt 
                                                                    # heating  J/m2s
        
        #if v > 4000:           # High speed turbulent eqs (v > 4km/s):
            # laminar flow, at distance1
        qWALL1 = Cthv1 * (rho ** 0.5) * (v ** 3.2) * (1 - (enthratioW1))
            # turbulent flow, high v, at distance2
        qWall2 = Cthv2 * (rho ** 0.5) * (v ** 3.2) * (1 - (enthratioW2))
        #else:                  # Low speed turbulent eqs (v <= 4km/s)
            # turbulent flow, low v, at distance1
        #    qWALL1 = Ctlv1 * (rho ** 0.8) * (v ** 3.37) * (1 - (1.11 * enthratioW1)) * ((556 / Tw1) ** 0.25)
            # turbulent flow, low v, at distance2
        #    qWall2 = Ctlv2 * (rho ** 0.8) * (v ** 3.37) * (1 - (1.11 * enthratioW2)) * ((556 / Tw2) ** 0.25)
        
        if qSP > 0 and qWALL1 > 0 and qWall2 > 0:       # heating eqs fail at low V
            # Calculate wall temperatures using Stephen-Boltzmann, assuming heat going 
            # in equals the heat radiated:
            Tsp = (qSP / (emis * sigma)) ** 0.25
            Tw1 = (qWALL1 / (emis * sigma)) ** 0.25
            Tw2 = (qWall2 / (emis * sigma)) ** 0.25
        
            # calculate IR emission over bands defined at start of program (lam1, 2, 3, 4)
            
            # Integrate both IR functions over wavelength band of interest and upper vehicle surface area
            # Returns tuples containing IR emission [0] and estimated error [1], in W/sr
            I_front_sw = integrate.dblquad(ir_em_front, noserad, len1, lambda x: lam1, lambda x: lam2, args=(h_p, c_l, k_b, em, len1, thetarad, phi1rad, phi2rad, Tw1))
            I_rear_sw = integrate.dblquad(ir_em_rear, len1, len2, lambda x: lam1, lambda x: lam2, args=(h_p, c_l, k_b, em, len1, thetarad, phi1rad, phi2rad, Tw2))
            
            IR_SBIRS = I_front_sw[0] + I_rear_sw[0]
            
            I_front_mw = integrate.dblquad(ir_em_front, noserad, len1, lambda x: lam3, lambda x: lam4, args=(h_p, c_l, k_b, em, len1, thetarad, phi1rad, phi2rad, Tw1))
            I_rear_mw = integrate.dblquad(ir_em_rear, len1, len2, lambda x: lam3, lambda x: lam4, args=(h_p, c_l, k_b, em, len1, thetarad, phi1rad, phi2rad, Tw2))
            
            IR_DSP = I_front_mw[0] + I_rear_mw[0]
            
        # print data points to output file:
        # x-coord (km), y coord (km), z coord (km), time (s), range along earth (km), altitude (km), 
        # speed (km/s), angle (deg), wall temp at stag point, wall temp at 1 m (K), 
        # wall temp at 3.5 m, over head radiant intensity in SWIR, in MWIR
        
        dat.write(f'{x/1000:<11.3f}   {y/1000:<11.3f}   {z/1000:<11.3f}   {t:<11.3f}   {range_earth/1000:<11.3f}   {crange_earth/1000:<11.3f}   {h/1000:<11.3f}  {pathlength/1000:<11.3f}  {v/1000:<11.3f}   '
                  f'{gammad:<11.3f}   {kappad:<11.3f}   {Tsp:<11.3f}   {Tw1:<11.3f}   {Tw2:<11.3f}   {IR_SBIRS/1000:<11.3f}   {IR_DSP/1000:<11.3f} \n')        

        tprint = tprint + dtprint           # set next time to print results
    
    # while loop stops once t exceeds tEND, range exceeds maxrange, or h < 0


# *****************END OF BIG LOOP OVER TIME*********************************

dat.close()

print(f'Flight time = {t:.1f} sec')
print(f'Flight time = {t/60:.1f} min')
print(f'Range = {range_earth/1000:.1f} km')
print(f'Cross range = {crange_earth/1000:.1f} km')