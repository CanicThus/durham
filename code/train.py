import time
from typing import Final
script_start: Final = time.perf_counter() # Do not remove or change this value
import torch
from UnifiedPuzzleSetZipDataset import make_unified_puzzle_dataloader_zip, batch_fragments_from_collate




# setup
# Todo: Add your code here

USERNAME="example_username"  # Replace with your username
MODEL_SAVE_PATH=f"{USERNAME}-assembly-model.pth"

# Data Loading
# Todo: Add your code here


# Model Definition
# Todo: Add your code here



# Training Loop
# Todo: Add your code here



# Save Model
def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    epoch: Optional[int] = None,
    best_val: Optional[float] = None,
    extra: Optional[Dict[str, Any]] = None,
):
    """
    Save a training checkpoint.

    Args:
        path: output .pt or .pth file
        model: model to save
        optimizer: optimizer (optional)
        scheduler: LR scheduler (optional)
        epoch: current epoch (optional)
        best_val: best validation loss so far (optional)
        extra: any additional metadata to store
    """
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None

    ckpt = {
        "model_state": model.state_dict(),
    }

    if optimizer is not None:
        ckpt["optimizer_state"] = optimizer.state_dict()
    if scheduler is not None:
        ckpt["scheduler_state"] = scheduler.state_dict()
    if epoch is not None:
        ckpt["epoch"] = epoch
    if best_val is not None:
        ckpt["best_val"] = best_val
    if extra is not None:
        ckpt["extra"] = extra

    torch.save(ckpt, path)
    print(f"Checkpoint saved to: {path}")


save_checkpoint(
    path=MODEL_SAVE_PATH,
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    epoch=EPOCHS,
    best_val=best_val,
)



# Qualitative Visualisation
# Optional Todo: Add your code here



# Count Number of Parameters
def count_trainable_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

num_params = count_trainable_parameters(model)
print(f"Trainable parameters: {num_params:,}")

# Hard safety check for oversized models
MAX_PARAMS = 15_000_000

if num_params > MAX_PARAMS:
    print("\n" + "!" * 80)
    print("WARNING: MODEL IS TOO LARGE")
    print(f"This model has {num_params:,} trainable parameters.")
    print(f"The recommended maximum for this assignment is {MAX_PARAMS:,}.")
    print()
    print("Large models:")
    print("- Train much more slowly")
    print("- Are more likely to overfit")
    print("- May exceed memory or runtime limits")
    print()
    print("Consider reducing:")
    print("- d_model")
    print("- number of Transformer layers")
    print("- number of attention heads")
    print("!" * 80 + "\n")


# Time Script
script_end = time.perf_counter()
total_time = script_end - script_start

print(f"Total execution time: {total_time:.6f} seconds")
