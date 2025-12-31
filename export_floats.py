from pathlib import Path
import re

def store_to_tex(label: str, value: float, tex_path: str = "./figures/values.tex"):
    """
    Store a label/value pair in a TeX file as a \newcommand.
    """
    # Remove all digits from label
    label_clean = re.sub(r"\d", "", label)

    tex_file = Path(tex_path)
    lines = tex_file.read_text().splitlines() if tex_file.exists() else []

    # Remove any line containing the cleaned label
    lines = [line for line in lines if label_clean not in line]

    # Add the new command
    lines.append(f"\\newcommand{{\\{label_clean}}}{{{value}}}")

    # Write back to file
    tex_file.write_text("\n".join(lines) + "\n")
