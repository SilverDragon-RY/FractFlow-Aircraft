import base64
from openai import OpenAI
from mcp.server.fastmcp import FastMCP
from typing import List
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize FastMCP server
mcp = FastMCP("gpt_image")

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

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

@mcp.tool()
async def edit_image_with_gpt(
    save_path: str,
    prompt: str,
    image_paths: List[str]
) -> str:
    """
    Edit and generate images using GPT's image editing capabilities with reference images.
    This tool takes multiple reference images and a text prompt to create a new image that
    combines elements from the reference images according to the prompt description.
    
    Args:
        save_path: Full path where the generated image will be saved (including filename)
        prompt: Text description of how to combine and modify the reference images
        image_paths: List of paths to reference images that will be used as input for editing
        
    Returns:
        Image file path as a string where the generated image is saved
        
    Example:
        To generate an image combining multiple character portraits:
        ```python
        result = await edit_image_with_gpt(
            save_path="output/group_photo.png",
            prompt="Three characters standing together in a group photo",
            image_paths=["char1.png", "char2.png", "char3.png"]
        )
        ```
    """

    model = "gpt-image-1"
    # Normalize the save path
    save_path = normalize_path(save_path)
    
    # Ensure the save directory exists
    save_dir = os.path.dirname(save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    
    # Open all reference images
    images = []
    for img_path in image_paths:
        normalized_path = normalize_path(img_path)
        with open(normalized_path, "rb") as f:
            images.append(f)
    
    try:
        # Generate image using GPT
        result = client.images.edit(
            model=model,
            image=images,
            prompt=prompt,
        )
        
        # Decode and save the generated image
        image_base64 = result.data[0].b64_json
        image_bytes = base64.b64decode(image_base64)
        
        with open(save_path, "wb") as f:
            f.write(image_bytes)
            
        return save_path
        
    finally:
        # Close all opened image files
        for img in images:
            img.close()

@mcp.tool()
async def create_image_with_gpt(
    save_path: str,
    prompt: str,
) -> str:
    """
    Generate a new image from scratch using GPT's image generation capabilities.
    This tool creates images based solely on a text prompt, without requiring any reference images.
    
    Args:
        save_path: Full path where the generated image will be saved (including filename)
        prompt: Detailed text description of the image to generate
        
    Returns:
        Image file path as a string where the generated image is saved
        
    Example:
        To generate a children's book style illustration:
        ```python
        result = await create_image_with_gpt(
            save_path="output/otter.png",
            prompt="A children's book drawing of a veterinarian using a stethoscope to listen to the heartbeat of a baby otter."
        )
        ```
    """
    model = "gpt-image-1"
    # Normalize the save path
    save_path = normalize_path(save_path)
    
    # Ensure the save directory exists
    save_dir = os.path.dirname(save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    
    try:
        # Generate image using GPT
        result = client.images.generate(
            model=model,
            prompt=prompt, 
            quality="low"
        )
        
        # Decode and save the generated image
        image_base64 = result.data[0].b64_json
        image_bytes = base64.b64decode(image_base64)
        
        with open(save_path, "wb") as f:
            f.write(image_bytes)
            
        return save_path
    except Exception as e:
        raise Exception(f"Failed to generate image: {str(e)}")
    finally:
        # Clean up any resources if needed
        pass

if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport='stdio') 