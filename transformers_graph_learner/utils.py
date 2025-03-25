# Helper function to move a dictionary of tensors to the device.
def to_device(data, device):
    data["tokens"] = data["tokens"].to(device)
    data["y"] = data["y"].to(device)
    data["node_count"] = data["node_count"].to(device)
    data["attn_mask"] = data["attn_mask"].to(device)
    return data