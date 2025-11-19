# Robot Self-Modeling AI Instructions

## Project Overview
This project implements a self-supervised learning framework for robots to model their own morphology and kinematics using visual data, based on the paper "Teaching Robots to Build Simulations of Themselves" (Nature Machine Intelligence, 2025). The core technique adapts Neural Radiance Fields (NeRF) to predict robot body occupancy conditioned on joint angles.

## Architecture & Components

### 1. Neural Network (Self-Model)
- **Core Logic**: Located in `data/SelfSim.ipynb`.
- **Model**: `FBV_SM` (Forward-Backward Visual Self-Model).
- **Input**: 3D coordinates (x, y, z) + Joint angles (motor commands).
- **Architecture**: Uses Positional Encoding (sine-cosine) followed by MLPs.
- **Framework**: PyTorch (`torch.nn.Module`).

### 2. Hardware Interface
- **Driver**: `RobotArmURDF/motor_babbling.py`.
- **Device**: LX16A Serial Bus Servos.
- **Communication**: Serial port (e.g., `/dev/ttyUSB0`), custom packet structure (Header `0x55`, ID, Length, Command, Parameters, Checksum).

### 3. Robot Definitions
- **Format**: URDF (Unified Robot Description Format).
- **Location**: `RobotArmURDF/` (e.g., `4dof_1st/urdf/4dof_1st.urdf`).
- **Variations**: Different configurations stored in subfolders (`4dof_1st`, `4dof_2nd`, etc.).

### 4. Data Management
- **Storage**: `data/sim_data/` contains `.npz` files.
- **Format**: NumPy archives with keys: `images`, `poses`, `angles`, `focal`.
- **Loading Pattern**:
  ```python
  data = np.load('data/sim_data/sim_data_robo1(ee).npz')
  images = data['images']
  angles = data['angles']
  ```

## Developer Workflows

### Training & Simulation
- **Primary Environment**: Jupyter Notebook (`data/SelfSim.ipynb`).
- **Device**: Code automatically selects CUDA if available.
- **Process**:
  1. Load `.npz` data.
  2. Initialize `FBV_SM` model and `PositionalEncoder`.
  3. Train using the training loop defined in the notebook.

### Hardware Control
- **Usage**: Import `LX16A` class from `motor_babbling.py`.
- **Pattern**:
  ```python
  from RobotArmURDF.motor_babbling import LX16A
  servo = LX16A(Port='COM3') # Adjust port as needed
  servo.moveServo(id=1, position=500)
  ```

## Conventions & Patterns
- **Robot IDs**: Robots are referenced by ID (e.g., `robo1`, `robo2`).
- **Coordinate Systems**: NeRF-style coordinate inputs for the model.
- **File Paths**: Use relative paths from the project root or the specific script's execution context.
- **Dependencies**: `torch`, `numpy`, `cv2`, `matplotlib`, `pyserial`.

## Key Files
- `data/SelfSim.ipynb`: Main entry point for model training and visualization.
- `RobotArmURDF/motor_babbling.py`: Low-level servo driver.
- `data/readme.md`: Documentation for data format and generation.
