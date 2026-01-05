# Robot Path Planning Simulation (Python)

## Project Overview
This project is a simple 2D simulation for robot navigation in an environment with randomly generated obstacles.  
The goal is to move a robot from a **start position** to a **goal position** while avoiding collisions and following the **shortest possible path**.

The focus of the project is understanding **path planning**, **collision avoidance**, and how the **A\*** algorithm works in practice.

---

## What the Project Does
- Generates **random geometric obstacles** (squares, rectangles, triangles)
- Obstacles are large and clearly visible, spread across the map
- The user defines the **start** and **goal** positions
- The robot computes an **optimal collision-free path**
- The environment and the path are displayed using a 2D plot

---

## Implementation Details
- Obstacles are modeled as **2D geometric shapes**
- To take the robot size into account, obstacles are expanded before planning
- The environment is discretized into a grid
- The **A\*** algorithm is used to find the shortest valid path
- The final result is visualized for easy interpretation

---

## Technologies Used
- **Python**
- **NumPy** – grid handling and numerical operations
- **Shapely** – geometric operations and collision checking
- **Matplotlib** – visualization of the environment and the path

---

## How to Run
Install the required libraries:

```bash
pip install numpy shapely matplotlib
