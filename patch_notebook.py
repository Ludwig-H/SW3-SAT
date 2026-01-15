import json

notebook_path = '/workspaces/SW3-SAT/sw3sat_colab.ipynb'
source_path = '/workspaces/SW3-SAT/tetra_class.py'

# Read the new source code
with open(source_path, 'r') as f:
    new_code = f.read()

# Convert to list of strings with \n (format expected by ipynb)
new_source_lines = []
lines = new_code.split('\n')
for i, line in enumerate(lines):
    if i < len(lines) - 1:
        new_source_lines.append(line + '\n')
    else:
        new_source_lines.append(line)

# Load notebook
with open(notebook_path, 'r') as f:
    nb = json.load(f)

# Find and replace
found = False
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source_joined = "".join(cell['source'])
        if "class TetraDynamicsGPU" in source_joined:
            cell['source'] = new_source_lines
            found = True
            break

if found:
    with open(notebook_path, 'w') as f:
        json.dump(nb, f, indent=1)
    print("Notebook updated successfully via external file patch.")
else:
    print("Target cell not found.")

