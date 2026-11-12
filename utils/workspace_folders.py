import os

# Base directory logic: the repo root, one level up from this file's own
# utils/ directory (this repo has no ~/cobra/curvature_calculation cluster path).
main_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Sub-folder definitions
#### READ-ONLY ####
geometry_inputs = os.path.join(main_folder, "geometry_inputs")
ML_data = os.path.join(main_folder, "ML_data")
#### NOT READ-ONLY ####
out_err_folder = os.path.join(main_folder, "out_err_files")
geometry_outputs = os.path.join(main_folder, "geometry_outputs")
ML_outputs = os.path.join(main_folder, "ML_outputs")
python_scripts = os.path.join(main_folder, "python_scripts")
python_outputs = os.path.join(main_folder, "python_outputs")
latex_notes = os.path.join(main_folder, "latex_notes")

# Task 31 energy-geodesic pilots (Phase 1+)
energy_geodesic_outputs = os.path.join(geometry_outputs, "energy_geodesic")
energy_geodesic_logs = os.path.join(out_err_folder, "energy_geodesic_logs")

# Ensure critical folders exist. Only out_err_folder is actually written to
# by anything that runs in this repo (chart_flow's Tee logger); the other
# constants above are kept because provenance-only scripts still import
# them, but nothing here creates directories for outputs this repo never
# produces.
os.makedirs(out_err_folder, exist_ok=True)
