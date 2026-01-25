import json

notebook_path = "Swendsen-Wang_3SAT_Colab.ipynb"
output_path = "generate_notebook.py"

with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

script_content = """import json

# Define the notebook structure
notebook = {
    "cells": [],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.8.5"
        },
        "colab": {
            "provenance": [],
            "gpuType": "A100"
        },
        "accelerator": "GPU"
    },
    "nbformat": 4,
    "nbformat_minor": 0
}

def add_markdown(source_string):
    lines = [line + "\n" for line in source_string.splitlines()]
    if lines: lines[-1] = lines[-1].rstrip("\n")
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {
            "id": "2KtKer9gg7Il"
        },
        "source": lines
    })

def add_code(source_string, execution_count=None, outputs=None):
    if outputs is None:
        outputs = []
    lines = [line + "\n" for line in source_string.splitlines()]
    if lines: lines[-1] = lines[-1].rstrip("\n")
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": execution_count,
        "metadata": {
            "id": "IIfZsdIYg7Iq" if "Environment" in source_string else None
        },
        "outputs": outputs,
        "source": lines
    })

# --- Content ---
"""

for i, cell in enumerate(nb["cells"]):
    cell_type = cell["cell_type"]
    source = "".join(cell["source"])
    
    # Use repr() to get a safe python string representation
    safe_source = repr(source)

    if cell_type == "markdown":
        script_content += f'\n# Cell {i} (Markdown)\n'
        script_content += f'md_source_{i} = {safe_source}\n'
        script_content += f'add_markdown(md_source_{i})\n'
    elif cell_type == "code":
        script_content += f'\n# Cell {i} (Code)\n'
        script_content += f'code_source_{i} = {safe_source}\n'
        
        outputs_repr = "[]"
        if cell.get("outputs"):
            cleaned_outputs = []
            for out in cell["outputs"]:
                if out.get("name") == "stdout":
                    cleaned_outputs.append({
                        "output_type": "stream",
                        "name": "stdout",
                        "text": out["text"]
                    })
            if cleaned_outputs:
                import json
                outputs_json = json.dumps(cleaned_outputs)
                outputs_repr = outputs_json
        
        script_content += f'add_code(code_source_{i}, execution_count=None, outputs={outputs_repr})\n'

script_content += """
with open("Swendsen-Wang_3SAT_Colab.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1)

print("Notebook generated successfully.")
"""

with open(output_path, "w", encoding="utf-8") as f:
    f.write(script_content)

print(f"Generator script created at {output_path}")