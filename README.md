
# alphaplane

`alphaplane` is an aircraft analysis and design library, currently in early development.

## Features

- **Airfoil Geometry Editing**
- **Airfoil Analysis**  
  - with [XFOIL](https://web.mit.edu/drela/Public/web/xfoil/xfoil_doc.txt) and [NeuralFoil](https://github.com/peterdsharpe/NeuralFoil)
- **Propeller Geometry Editing**
- **Propeller Analysis**  
  - implementation of [QPROP](https://web.mit.edu/drela/Public/web/qprop/) which integrates with calculated airfoil data
  - also supports given CT, CQ, and RPM data
- **Brushless Motor Analysis**  
  - 3-constant (Kv, I0, R) motor model
- **Full Electric Propulsion Analysis**  
  - Analyze setup with battery, speed controller, motor, and propeller
- **Database**  
  - Includes a database of airfoils (from Selig), propellers (from APC), and motors (from Scorpion)

## Installation

Requires python >= 3.11

Currently only works on Windows

1. **Install neuralfoil-standalone**  
   Get `neuralfoil-standalone` from [this repository](https://github.com/pedro-crb/neuralfoil-standalone) and follow the installation instructions.

2. **Download or Clone the Repository**  
   Clone the repository or download it directly from the GitHub page

3. **Install the Package**  
   ```bash
   pip install path/to/alphaplane
   ```
   `path/to/alphaplane` should be the folder containing `setup.py`.

## Documentation

*Currently work in progress...*

Check out the `examples` folder for some use cases

## Future Plans

- Documentation
- 3D wing geometry
- 3D wing aerodynamics using vortex panels
- Propeller analysis using lifting line theory
- general 3D body using vortex panels
- ducted propeller analysis
- aircraft geometry
- stability and control analysis (stability derivatives / flight dynamics)
- performance analysis (takeoff / steady turn)
- . . .
