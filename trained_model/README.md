# Trained Model Directory

This directory stores checkpoint files produced by notebook experiments.

Typical contents:

- KAN parameter snapshots at selected time values
- intermediate checkpoints used for visualization or comparison

Notes:

- file names generally encode the model family and the corresponding time
- not every notebook requires every checkpoint in this directory
- additional runs may overwrite or add new `.pt` files if the notebook save cells are executed

