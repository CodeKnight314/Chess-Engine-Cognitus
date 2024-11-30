# Chess-Engine-Cognitus

## Overview
<p align="center">
  <img src="images/Cognitus Logo.png" alt="Cognitus Logo">
</p>

This is a python-based chess engine for deploying algorithmic-based min-maxing game-playing agents with Alpha-Beta pruning. Due to Python's inherent slow runtime, the engine, Cognitus, is restricted to limited depth and time. As of now, it performs suboptimally at Depth 3-5 for any given board with a 20-second time limit. With future optimization, Cognitus aims to run at near-competitive speeds and search at comparable depths.

The project's previous focus on ML and Deep-Q-Learning based chess engine has been moved to a separate repository due to a change of this repository's focus and computation budget issues.

## Saved games 
`game.py` saves all played games as pgn files in `saved_pgn`. Games can be reviewed via chess.com's pgn evaluator for free.

## Performance

* **Elo Rating:** Estimated to be between 1550 and 1700.
* **Nodes per Second (NPS):** Averages around 2800-4000.
* **Search Depth:** Currently limited to 3-5 ply due to performance constraints.
* **Playing Style:** No distinct playing style has emerged yet.

## Shortcomings

* **Known Bugs:** User-computer games do not run smoothly, with potential delays or unresponsiveness.
* **Areas for Improvement:** 
    * Overall speed and efficiency need improvement.
    * Search depth is limited.
    * Evaluation heuristics can be refined. 

## To-Do List of Optimizations

* **Implement Cython/Numba for speed improvements:** This will significantly boost the engine's calculation speed, allowing for deeper searches and faster response times.
* **Increase Search Depth:**  Aim to increase the search depth to at least 8-10 ply to improve the engine's strategic planning and tactical awareness.
* **Implement Monte Carlo Search Tree (MCTS):**  Explore the use of MCTS for a more robust and potentially creative playing style.
* **Refine Evaluation Heuristics:**  Improve the evaluation function to better assess positional advantages, material imbalances, and other critical factors.

## How to Use via Command Line

**Installation:**

1. Clone the repository: `git clone https://github.com/CodeKnight314/Chess-Engine-Cognitus.git`
2. Navigate to the directory: `cd Chess-Engine-Cognitus`
3. Install necessary libraries: `pip install -r requirement.txt`

**Basic Usage:**
To run a game between the computer and itself, cd to the project folder and run the following command:
```bash
python game.py
```
