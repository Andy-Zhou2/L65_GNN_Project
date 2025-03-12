# Helper function to move a dictionary of tensors to the device.
def to_device(data, device):
    data["tokens"] = data["tokens"].to(device)
    data["y"] = data["y"].to(device)
    return data