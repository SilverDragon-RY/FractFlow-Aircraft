import os
import asyncio
from typing import Any
from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv
import replicate
import httpx

# Load environment variables
load_dotenv()

# Initialize FastMCP server
mcp = FastMCP("sam")

def normalize_path(path: str) -> str:
    """
    Normalize a file path by expanding ~ to user's home directory
    and resolving relative paths.
    
    Args:
        path: The input path to normalize
        
    Returns:
        The normalized absolute path
    """
    # Expand ~ to user's home directory
    expanded_path = os.path.expanduser(path)
    
    # Convert to absolute path if relative
    if not os.path.isabs(expanded_path):
        expanded_path = os.path.abspath(expanded_path)
        
    return expanded_path

async def download_file(url: str, save_path: str) -> None:
    """
    Download a file from URL to local path.
    
    Args:
        url: The URL to download from
        save_path: Local path to save the file
    """
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        response.raise_for_status()
        
        with open(save_path, 'wb') as f:
            f.write(response.content)

@mcp.tool()
async def segment_anything_v2(
    image_path: str,
    save_directory: str = "",
    points_per_side: int = 32,
    pred_iou_thresh: float = 0.88,
    stability_score_thresh: float = 0.95
) -> str:
    """
    Segment objects in an image using SAM 2 (Segment Anything v2) model.
    
    This tool uses the meta/sam-2 model on Replicate to generate object masks
    for the input image. It returns both a combined mask and individual masks
    for each detected object.
    
    Args:
        image_path (str): Path to the input image file
        save_directory (str): Directory where output masks will be saved
        points_per_side (int): Number of points per side for mask generation (default: 32)
        pred_iou_thresh (float): Predicted IOU threshold (default: 0.88)
        stability_score_thresh (float): Stability score threshold (default: 0.95)
    
    Returns:
        str: Success message with information about saved files
    """
    try:
        # Normalize image path
        image_path = normalize_path(image_path)
        
        # Auto-generate save directory if not provided
        if not save_directory.strip():
            image_dir = os.path.dirname(image_path)
            image_name = os.path.splitext(os.path.basename(image_path))[0]
            save_directory = os.path.join(image_dir, f"{image_name}_SAM_MASKS")
        else:
            save_directory = normalize_path(save_directory)
        
        # Ensure save directory exists
        os.makedirs(save_directory, exist_ok=True)
        
        # Initialize Replicate client
        client = replicate.Client(api_token=os.getenv('REPLICATE_API_TOKEN'))
        
        # Run the SAM 2 model
        output = client.run(
            "meta/sam-2:fe97b453a6455861e3bac769b441ca1f1086110da7466dbb65cf1eecfd60dc83",
            input={
                "image": open(image_path, "rb"),
                "points_per_side": points_per_side,
                "pred_iou_thresh": pred_iou_thresh,
                "stability_score_thresh": stability_score_thresh
            }
        )
        
        # Download combined mask
        combined_mask_path = os.path.join(save_directory, "combined_mask.png")
        await download_file(str(output["combined_mask"]), combined_mask_path)
        
        # Download individual masks
        individual_mask_paths = []
        for i, mask_file in enumerate(output["individual_masks"]):
            mask_path = os.path.join(save_directory, f"mask_{i}.png")
            await download_file(str(mask_file), mask_path)
            individual_mask_paths.append(mask_path)
        
        # Return specific file paths
        result = f"Segmentation completed!\nCombined mask: {combined_mask_path}\nIndividual masks ({len(individual_mask_paths)} files):\n"
        for mask_path in individual_mask_paths:
            result += f"- {mask_path}\n"
        
        return result.strip()
        
    except Exception as e:
        return f"Error during segmentation: {str(e)}"

if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport='stdio') 