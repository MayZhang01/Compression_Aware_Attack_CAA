import torch

def generate_patch_mask(selected_indices, image_size=336, patch_size=14):
    """
    Generate a (image_size, image_size) mask where pixels corresponding to selected patch tokens are set to 1,
    and all other pixels are 0.

    Args:
        selected_indices (torch.Tensor): 1D tensor containing the indices of selected tokens (values in [0, num_patches-1]).
        image_size (int): Size of the input image (height and width). Default is 336.
        patch_size (int): Size of each patch (height and width). Default is 14.

    Returns:
        torch.Tensor: A 2D tensor mask of shape (image_size, image_size) with 1s in selected regions.
    """
    num_patches_per_row = image_size // patch_size

    # Initialize mask with zeros
    mask = torch.zeros((image_size, image_size), dtype=torch.float32)

    # Iterate over each selected token index
    for idx in selected_indices:
        # Compute the row and column index of the patch
        row = idx // num_patches_per_row
        col = idx % num_patches_per_row

        # Compute the pixel coordinates of the patch
        row_start = row * patch_size
        row_end = (row + 1) * patch_size
        col_start = col * patch_size
        col_end = (col + 1) * patch_size

        # Set the corresponding region in the mask to 1
        mask[row_start:row_end, col_start:col_end] = 1.0

    return mask

# Example usage:
if __name__ == "__main__":
    selected_indices = torch.tensor([25, 100, 300])
    mask = generate_patch_mask(selected_indices)
    print(mask.shape)  # Should be (336, 336)