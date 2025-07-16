import sys
import torch
def get_size_in_bytes(obj):
    if isinstance(obj, dict):
        size = 0
        for k, v in obj.items():
            size += get_size_in_bytes(k)
            size += get_size_in_bytes(v)
        return size
    elif isinstance(obj, torch.Tensor):
        return obj.element_size() * obj.nelement()
    elif isinstance(obj, (list, tuple, set)):
        return sum(get_size_in_bytes(i) for i in obj)
    else:
        # fallback for other Python objects
        return sys.getsizeof(obj)

def get_size_in_gb(obj):
    size_bytes = get_size_in_bytes(obj)
    return size_bytes / (1024 ** 3)
