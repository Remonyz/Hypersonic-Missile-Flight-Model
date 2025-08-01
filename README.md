# Hypersonic Missile Flight Model
https://hypersonic-missile-flight-model.onrender.com/

A comprehensive physics-based simulation system for analyzing hypersonic glide vehicle trajectories, thermal heating, and infrared emissions. This project includes both standalone Python scripts and an interactive web interface for advanced trajectory analysis and footprint calculations.

## Features

- **6DOF Trajectory Simulation**: Full six-degree-of-freedom equations of motion for hypersonic flight
- **Atmospheric Modeling**: Accurate atmospheric density calculations up to 700km altitude using Regan's atmospheric model
- **Thermal Analysis**: Aerodynamic heating calculations for both laminar and turbulent flow regimes
- **IR Emission Modeling**: Infrared signature calculations in SWIR and MWIR wavelength bands using Planck's law
- **Footprint Analysis**: Threatened area calculations with convex hull algorithms
- **Interactive Web Interface**: Real-time visualization with trajectory plotting and parameter adjustment
- **Geographic Projection**: Map-based trajectory visualization with Earth coordinate systems
- **Complex Maneuvers**: Support for time-varying roll angle changes during flight

## Project Structure

```
Hypersonic-Missile-Flight-Model/
├── Hyper flight 9_23_22 laminar.py    # Standalone laminar flow simulation
├── Hyper flight 9_23_22 turbulent.py  # Standalone turbulent flow simulation
├── server.py                           # Flask web server with simulation engine
├── templates/
│   └── index.html                      # Interactive web interface
├── requirements.txt                    # Python dependencies
└── README.md                          # This file
```

### File Descriptions

- **`Hyper flight 9_23_22 laminar.py`**: Original Python conversion from BASIC implementing laminar flow heating models for HTV-2 style vehicles
- **`Hyper flight 9_23_22 turbulent.py`**: Enhanced version with turbulent flow heating calculations and velocity-dependent flow regime selection
- **`server.py`**: Flask web application providing REST API endpoints and the main `HypersonicGlideSimulator` class
- **`templates/index.html`**: Full-featured web interface with parameter controls, real-time plotting, and map integration

## Installation

### Prerequisites

- Python 3.7 or higher
- Modern web browser (for web interface)

### Setup

1. Clone or download this repository
2. Install required dependencies:

```bash
pip install -r requirements.txt
```

### Dependencies

- **Flask**: Web framework for the interactive interface
- **NumPy**: Numerical computations and array operations
- **SciPy**: Integration functions and scientific computing
- **Matplotlib**: Plot generation for standalone scripts
- **Plotly**: Interactive web-based plotting (loaded via CDN)

## Usage

### Web Interface (Recommended)

1. Start the Flask server:

```bash
python server.py
```

2. Open your web browser and navigate to:
```
http://localhost:5051
```

3. Use the interactive interface to:
   - Adjust initial flight conditions (velocity, angles)
   - Configure vehicle properties (mass, ballistic coefficient, L/D ratio)
   - Set up complex maneuvers with time-varying roll angles
   - Run trajectory simulations
   - Calculate threatened footprints
   - Visualize results on interactive plots and maps

### Standalone Scripts

Run either simulation script directly:

```bash
python "Hyper flight 9_23_22 laminar.py"
```

or

```bash
python "Hyper flight 9_23_22 turbulent.py"
```

These scripts will generate output files:
- `HTV2_flight-Lat-Long-R&XR.dat`: Trajectory data
- `HTV2_flight-Lat-Long-R&XR.prm`: Simulation parameters

## Parameters

### Initial Flight Conditions

- **Velocity (v0)**: Initial speed in km/s (typically 3-8 km/s for hypersonic flight)
- **Pitch Angle (gamma0)**: Angle from horizontal in degrees (negative = descending)
- **Roll Angle (roll0)**: Vehicle orientation angle in degrees
- **Heading Angle (kappa0)**: Direction angle measured from north in degrees

### Vehicle Properties

- **Mass (payload)**: Vehicle mass in kg
- **Ballistic Coefficient (beta)**: β = m/(C_D × A) in kg/m²
- **Lift-to-Drag Ratio (L/D)**: Aerodynamic efficiency parameter
- **Surface Emissivity**: Thermal radiation parameter (0.1-1.0)

### Coordinate System

The simulation uses a spherical coordinate system where:
- **PSI (ψ)**: Longitude angle (range from launch point)
- **OMEGA (ω)**: Latitude angle (crossrange from launch point)  
- **KAPPA (κ)**: Heading angle measured from north
- **Origin**: Center of Earth
- **Earth Radius**: 6,370 km

## Technical Details

### Physics Models

The simulation implements several key physics models:

1. **Equations of Motion**: Modified 6DOF equations for hypersonic glide vehicles
2. **Atmospheric Model**: Frank J. Regan's atmospheric density model (1984)
3. **Heating Calculations**: Anderson's hypersonic heating equations
4. **IR Emissions**: Planck's law for thermal radiation in specified wavelength bands

### Integration Method

- **Algorithm**: Modified Runge-Kutta (midpoint method) for improved stability
- **Time Step**: Adaptive from 0.1s initial to 0.1s for t > 1s
- **Termination**: Flight ends when altitude < 0, time > 3600s, or range > 20,000 km

### Flow Regimes

The turbulent version automatically selects heating models based on velocity:
- **High Speed (v > 4 km/s)**: Turbulent flow equations with v^3.7 dependency
- **Low Speed (v ≤ 4 km/s)**: Modified turbulent equations with v^3.37 dependency

## Output Data

### Trajectory Files (.dat)

Generated data includes:
- Position coordinates (x, y, z) in km
- Time stamps in seconds
- Range and crossrange along Earth surface in km
- Altitude above Earth in km
- Velocity magnitude in km/s
- Flight path angles in degrees
- Surface temperatures at multiple points in Kelvin
- Infrared radiant intensity in kW/sr for different wavelength bands

### Parameter Files (.prm)

Records simulation settings:
- Initial conditions
- Vehicle parameters
- Integration settings
- Heating calculation coefficients

## Contributing

Contributions are welcome! Please follow these guidelines:

1. Maintain the existing code structure and naming conventions
2. Add appropriate comments for new physics models
3. Test both standalone scripts and web interface functionality
4. Update documentation for new parameters or features

## References and Citations

This project is based on:

1. **Original BASIC Code**: HYP-HT-2 ballistic trajectory program
2. **Anderson, J.D.**: "Hypersonic and High Temperature Gas Dynamics", McGraw-Hill
3. **Regan, F.J.**: "Re-entry Vehicle Dynamics", MIT Press (1984)
4. **Vehicle Geometry**: Based on HTV-2 (Hypersonic Technology Vehicle 2) specifications

### Key Equations

- **Heating**: Anderson's stagnation point and wall heating equations
- **Atmospheric Density**: Regan's model with temperature and pressure variations
- **IR Emissions**: Planck's law integrated over specified wavelength bands
- **Trajectory**: Spherical coordinate system with Earth rotation effects

## License

This project is provided for educational and research purposes. Please cite the original sources and this implementation when used in academic work.

## Troubleshooting

### Common Issues

1. **Port Already in Use**: Change the port in `server.py` (line 950)
2. **Missing Dependencies**: Ensure all packages in `requirements.txt` are installed
3. **Large Output Files**: Adjust `dtprint` parameter to reduce data density
4. **Simulation Crashes**: Check initial conditions are physically reasonable

### Performance Notes

- Web interface is optimized for typical desktop browsers
- Large footprint calculations may take 30-60 seconds
- Standalone scripts are faster for batch processing
- Integration time step affects both accuracy and runtime

## Version History

- **Current**: Python 3.x with Flask web interface
- **Original**: BASIC implementation for ballistic calculations
- **Enhanced**: Added thermal analysis and IR emission modeling
