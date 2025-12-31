import re
import numpy as np
def rename_model(x):
    if x == "OLMo-2-0425-1B_tulu-3-sft-olmo-2-mixture-0225_lr0.0001_seed42":
            return "Olmo2-1B"
    if x == "Qwen2.5-0.5B_tulu-3-sft-olmo-2-mixture-0225_lr0.0001_seed42":
                return "Qwen2.5-0.5B"
    if x == "Llama-3.2-1B_tulu-3-sft-olmo-2-mixture-0225_lr0.0001_seed42":
                return "Llama-3.2-1B"
       
def rename_estimator(x):
      return x.split(":")[0]
def rename_linear_coder(x):
      return x.replace("Coder","").replace("Thresh","")
def extract_seed(x):
       return int(re.search(r'seed (\d+)', x).group(1)) if "seed" in x else None

def rename_random(x):
    return re.sub(r' with seed \d+', "", x)

def rename_explanation_type(x, include_k=True):
    x = x.replace("X", "2202")
    if "sanity check" in x.lower():
        return "Test instance"

    # "<number> random examples with seed <number>"
    m = re.match(r"(\d+) random examples with seed (\d+)", x)
    if m:
        return f"{m.group(1)} rand" if include_k else "rand"
    elif "random" in x:
        return "random"

    # "Top-k (least|most) (influential|helpful|harmful)"
    m = re.match(r"Top-(\d+) (least|most) (influential|helpful|harmful).*", x)
    if m:
        k, direction, kind = m.groups()
        abbrev = f"{k} {direction} {kind}" if include_k else f"{direction} {kind}"
        return abbrev.replace("influential", "inf.")

    # "<num> by facility location from Top-<num> ... lambda=<num>" OR "Top-<num> by facility location ..."
    m = re.match(
        r"(?:(\d+) by|Top-(\d+) by) facility location from Top-\d+ (least|most) (influential|helpful|harmful)(?:.*lambda=([\d.]+))?",
        x
    )
    if m:
        old_num, new_num, direction, kind, lam = m.groups()
        num = old_num or new_num
        if lam:
            if lam == "0.0":
                lam_clean = "0"
            else:
                lam_clean = lam.lstrip("0") if lam.startswith("0") else lam
                if lam_clean.endswith(".0"):
                    lam_clean = lam_clean[:-2]
            lam_str = f" $\\lambda={lam_clean}$"
        else:
            lam_str = ""
        prefix = f"{num} by" if include_k else ""
        abbrev = f"{prefix} FL {direction} {kind}{lam_str}"
        return abbrev.replace("influential", "inf.")
    
    
    
    # "<num> by DIVINE from Top-<num> ... lambda=<num>"
    m = re.match(
        r"(?:(\d+) by|Top-(\d+) by) DIVINE from Top-\d+ (least|most) (influential|helpful|harmful)",
        x
    )
    if m:
        old_num, new_num, direction, kind = m.groups()
        num = old_num or new_num
        prefix = f"{num} by" if include_k else ""
        abbrev = f"{prefix} DIVINE {direction} {kind}"
        return abbrev.replace("influential", "inf.")

    # "<num> by AIDE from Top-<num>."
    m = re.match(
        r"(?:(\d+) by|Top-(\d+) by) AIDE from Top-\d+",
        x
    )
    if m:
        old_num, new_num = m.groups()
        num = old_num or new_num
        prefix = f"{num} by" if include_k else ""
        abbrev = f"{prefix} AIDE"
        return abbrev.replace("influential", "inf.")

    return x[:30] + "..." if len(x) > 30 else x

def extract_k(explanation_type):
    # Case: "The test instance (as a sanity check)"
    if "The test instance (as a sanity check)" in explanation_type:
        return 1

    # Case: "<number> by ... from Top-..."
    first_number_match = re.match(r"^\s*(\d+)\b", explanation_type)
    if first_number_match:
        return int(first_number_match.group(1))

    # Case: "Top-<number> ..." when no leading number
    top_match = re.search(r"Top-(\d+)", explanation_type)
    if top_match:
        return int(top_match.group(1))

    # Case: "<number> random examples"
    random_match = re.search(r"(\d+)\s+random examples", explanation_type)
    if random_match:
        return int(random_match.group(1))

    return None

def replace_k(explanation_type, k):
    if k is None:
        return explanation_type
    # Only replace the first occurrence of the number as a standalone word
    return re.sub(rf"\b{k}\b", "X", explanation_type, count=1)

def vectorized_replace_k(explanation_types, ks):
    result = explanation_types.copy()
    for k in np.unique(ks[ks.notnull()]):  # only unique, non-null ks
        # Use string pattern, not compiled regex
        pattern = rf"\b{k}\b"
        mask = ks == k
        result.loc[mask] = result.loc[mask].str.replace(pattern, "X", n=1, regex=True)
    return result

def facility_location_hotfix(x):
    if ("facility" in x) and x.startswith("Top-"):
        return x[len("Top-"):]
    else:
        return x
def get_sort_type(x):
    for sort_type in ["scores with largest absolute value", "most positive scores", "most negative scores", "scores closest to zero"]:
        if sort_type in x:
            return sort_type
    return "-"


