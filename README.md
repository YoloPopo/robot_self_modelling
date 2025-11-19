# Robot Self-Modeling Project

This project implements the paper **"Teaching Robots to Build Simulations of Themselves"** (Nature Machine Intelligence, 2025).

## The Goal
We want to create a robot that can "imagine" itself. Instead of programming the robot with a file that says "your arm is 20cm long," we want the robot to look at itself through a camera, move around, and learn its own shape and how it moves.

## How It Works
1.  **Data Collection**: The robot moves its arm and takes pictures.
2.  **Learning**: A neural network (AI) looks at the pictures and the motor angles. It tries to guess the shape of the robot.
3.  **Self-Model**: After training, the robot has a "mental model" of itself. It can predict where its body will be for any movement, without needing a physical simulation file.

## Project Status

### ✅ What is Done
-   **The AI Model**: We built the neural network (FBV_SM) using PyTorch. It works correctly.
-   **Training System**: We have a working training loop that learns from data.
-   **Validation**: The system checks itself to make sure it is learning correctly.
-   **Visualization**: We can see what the robot is imagining.
-   **Synthetic Test**: Since our original data was broken, we built a "stick figure" simulator to prove our code works. It successfully learned the stick figure robot.

### 🚧 What Remains
-   **Realistic Simulation**: We are moving from stick figures to a realistic 3D robot simulation using **PyBullet**.
-   **Lorenz Trajectories**: To gather better data, we will use the **Lorenz System** (a chaotic math equation) to generate smooth, complex movements for the robot arm. This replaces simple random movements.

## Repository Structure
-   `data/SelfSim.ipynb`: The main brain. This notebook trains the AI and shows the results.
-   `RobotArmURDF/`: Files that describe the physical robot (for the simulator).
-   `INTERMEDIATE_REPORT.md`: A detailed story of our progress so far.

## How to Run
1.  Install the requirements: `pip install torch numpy opencv-python matplotlib`
2.  Open `data/SelfSim.ipynb`.
3.  Run the cells. It will generate synthetic data and train the model before your eyes.