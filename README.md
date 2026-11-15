# BIT 2118 Scientific Computing - Individual Submission

Student: Eric Muli
Registration number: BSCCS202457153

This repository contains my individual work for the BIT 2118 group tasks (Tasks 2 to 20).
Discussions were done as a group; this repository and the documentation are my own.

## Structure

- `code/` - the original program for every task (Task_02 ... Task_20) and `parameter_changes.py`,
  which changes at least one input per task and prints the new results (Instruction C).
- `extensions/` - one modified program per task that solves a different but related real-world
  case (Instruction E). Assumptions are written at the top of every file.
- `outputs/` - text output captured from running every program.
- `figures/` - graphs produced by the programs (black and white).
- `docs/` - the Word documentation (assumptions, inputs, outputs, interpretation).
- `requirements.txt` - Python libraries needed.

## How to run


pip install -r requirements.txt
python code/Task_02_numpy_arrays.py
python code/parameter_changes.py
python extensions/Ext_02_vaccine_fridge.py


Run each command from the repository root so that graphs are saved in `figures/`.
Timing results (Task 3) vary from computer to computer.
