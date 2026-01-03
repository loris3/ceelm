from pathlib import Path
import re

def store_to_tex(label: str, value: float, tex_path: str = "./figures/values.tex"):
    """
    Store a label/value pair in a TeX file as a \\newcommand.
    """

    label_clean = re.sub(r"\d", "", label)
    tex_file = Path(tex_path)
    tex_file.parent.mkdir(parents=True, exist_ok=True)
    lines = tex_file.read_text().splitlines() if tex_file.exists() else []
    lines = [line for line in lines if label_clean not in line]
  
    lines.append(f"\\newcommand{{\\{label_clean}}}{{{value}}}")
    tex_file.write_text("\n".join(lines) + "\n")
