import os
import asyncio
from typing import Any, Optional, List
from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv
import replicate
import httpx
from urllib.parse import urlparse
import json
from PIL import Image

# Load environment variables
load_dotenv()

# Initialize FastMCP server
mcp = FastMCP("grounding_dino")

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
async def detect_objects_with_grounding_dino(
    image_path: str,
    query: str,
    box_threshold: float = 0.35,
    text_threshold: float = 0.25,
    show_visualisation: bool = True,
    save_directory: str = ""
) -> str:
    """
    Detect objects in an image using Grounding DINO model with text queries.
    
    This tool uses the adirik/grounding-dino model on Replicate to detect and locate
    objects in images based on natural language descriptions. It can detect arbitrary
    objects described in text format.
    
    Args:
        image_path (str): Path to the input image file
        query (str): Comma-separated text descriptions of objects to detect (e.g., "person, car, dog")
        box_threshold (float): Confidence level for object detection (default: 0.35)
        text_threshold (float): Confidence level for text matching (default: 0.25)
        show_visualisation (bool): Whether to generate and save annotated image with bounding boxes (default: True)
        save_directory (str): Directory where output files will be saved (optional)
    
    Returns:
        str: Detection results with bounding boxes, confidence scores, and file paths
    """
    try:
        # Normalize image path
        image_path = normalize_path(image_path)
        
        # Validate image file exists
        if not os.path.exists(image_path):
            return f"Error: Image file not found at {image_path}"
        
        # Auto-generate save directory if not provided
        if not save_directory.strip():
            image_dir = os.path.dirname(image_path)
            image_name = os.path.splitext(os.path.basename(image_path))[0]
            save_directory = os.path.join(image_dir, f"{image_name}_GROUNDING_DINO_RESULTS")
        else:
            save_directory = normalize_path(save_directory)
        
        # Ensure save directory exists
        os.makedirs(save_directory, exist_ok=True)
        
        # Initialize Replicate client
        client = replicate.Client(api_token=os.getenv('REPLICATE_API_TOKEN'))
        
        # Run the Grounding DINO model
        output = client.run(
            "adirik/grounding-dino:efd10a8ddc57ea28773327e881ce95e20cc1d734c589f7dd01d2036921ed78aa",
            input={
                "image": open(image_path, "rb"),
                "query": query,
                "box_threshold": box_threshold,
                "text_threshold": text_threshold,
                "show_visualisation": show_visualisation
            }
        )
        
        # Prepare serializable results
        serializable_output = {
            "detections": output.get("detections", []),
            "result_image_url": str(output.get("result_image", "")) if output.get("result_image") else None
        }
        
        # Save detection results to JSON file
        results_file = os.path.join(save_directory, "detection_results.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_output, f, indent=2, ensure_ascii=False)
        
        # Download annotated image if visualization is enabled
        annotated_image_path = None
        if show_visualisation and output.get("result_image"):
            annotated_image_path = os.path.join(save_directory, "annotated_image.png")
            await download_file(str(output["result_image"]), annotated_image_path)
        
        # Format detection results
        detections = output.get("detections", [])
        result_text = f"Object Detection Results for query: '{query}'\n"
        result_text += f"Total objects detected: {len(detections)}\n\n"
        
        if detections:
            result_text += "Detected Objects:\n"
            for i, detection in enumerate(detections, 1):
                bbox = detection.get("bbox", [])
                label = detection.get("label", "unknown")
                confidence = detection.get("confidence", 0.0)
                
                result_text += f"{i}. {label}\n"
                result_text += f"   Confidence: {confidence:.3f}\n"
                result_text += f"   Bounding Box: [x1={bbox[0]}, y1={bbox[1]}, x2={bbox[2]}, y2={bbox[3]}]\n\n"
        else:
            result_text += "No objects detected matching the query.\n\n"
        
        # Add file information
        result_text += f"Files saved to: {save_directory}\n"
        result_text += f"- Detection results: {results_file}\n"
        if annotated_image_path:
            result_text += f"- Annotated image: {annotated_image_path}\n"
        
        return result_text.strip()
        
    except Exception as e:
        return f"Error during object detection: {str(e)}"

@mcp.tool()
async def detect_and_crop_objects(
    image_path: str,
    query: str,
    box_threshold: float = 0.35,
    text_threshold: float = 0.25,
    save_directory: str = "",
    padding: int = 10
) -> str:
    """
    Detect objects in an image and crop them out as separate images.
    
    This tool first detects objects using Grounding DINO, then crops each detected 
    object from the original image and saves them as individual files.
    
    Args:
        image_path (str): Path to the input image file
        query (str): Comma-separated text descriptions of objects to detect
        box_threshold (float): Confidence level for object detection (default: 0.35)
        text_threshold (float): Confidence level for text matching (default: 0.25)
        save_directory (str): Directory where cropped images will be saved (optional)
        padding (int): Extra pixels to add around each crop (default: 10)
    
    Returns:
        str: JSON string containing detection results and list of cropped image paths
    """
    try:
        # Normalize image path
        image_path = normalize_path(image_path)
        
        # Validate image file exists
        if not os.path.exists(image_path):
            return json.dumps({
                "error": f"Image file not found at {image_path}",
                "cropped_images": []
            })
        
        # Auto-generate save directory if not provided
        if not save_directory.strip():
            image_dir = os.path.dirname(image_path)
            image_name = os.path.splitext(os.path.basename(image_path))[0]
            save_directory = os.path.join(image_dir, f"{image_name}_CROPPED_OBJECTS")
        else:
            save_directory = normalize_path(save_directory)
        
        # Ensure save directory exists
        os.makedirs(save_directory, exist_ok=True)
        
        # Initialize Replicate client
        client = replicate.Client(api_token=os.getenv('REPLICATE_API_TOKEN'))
        
        # Run the Grounding DINO model
        output = client.run(
            "adirik/grounding-dino:efd10a8ddc57ea28773327e881ce95e20cc1d734c589f7dd01d2036921ed78aa",
            input={
                "image": open(image_path, "rb"),
                "query": query,
                "box_threshold": box_threshold,
                "text_threshold": text_threshold,
                "show_visualisation": False  # We don't need the annotated image for cropping
            }
        )
        
        # Get detections
        detections = output.get("detections", [])
        
        if not detections:
            return json.dumps({
                "message": "No objects detected matching the query",
                "query": query,
                "total_detections": 0,
                "cropped_images": []
            })
        
        # Open the original image for cropping
        original_image = Image.open(image_path)
        image_width, image_height = original_image.size
        
        cropped_image_paths = []
        detection_results = []
        
        # Process each detection
        for i, detection in enumerate(detections, 1):
            bbox = detection.get("bbox", [])
            label = detection.get("label", "unknown")
            confidence = detection.get("confidence", 0.0)
            
            if len(bbox) != 4:
                continue
                
            x1, y1, x2, y2 = bbox
            
            # Add padding and ensure coordinates are within image bounds
            x1 = max(0, int(x1) - padding)
            y1 = max(0, int(y1) - padding)
            x2 = min(image_width, int(x2) + padding)
            y2 = min(image_height, int(y2) + padding)
            
            # Crop the object
            cropped_image = original_image.crop((x1, y1, x2, y2))
            
            # Generate filename for cropped image
            safe_label = label.replace(" ", "_").replace("/", "_").replace("\\", "_")
            crop_filename = f"{safe_label}_{i}_conf{confidence:.2f}.png"
            crop_path = os.path.join(save_directory, crop_filename)
            
            # Save cropped image
            cropped_image.save(crop_path)
            cropped_image_paths.append(crop_path)
            
            # Add detection info
            detection_results.append({
                "index": i,
                "label": label,
                "confidence": confidence,
                "bbox": bbox,
                "cropped_image_path": crop_path,
                "crop_size": [x2 - x1, y2 - y1]
            })
        
        # Save detection results with cropped paths
        results_file = os.path.join(save_directory, "detection_and_crop_results.json")
        full_results = {
            "query": query,
            "total_detections": len(detections),
            "detections": detection_results,
            "cropped_images": cropped_image_paths,
            "save_directory": save_directory,
            "original_image": image_path
        }
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(full_results, f, indent=2, ensure_ascii=False)
        
        # Return summary as JSON string
        return json.dumps({
            "message": f"Successfully detected and cropped {len(detections)} objects",
            "query": query,
            "total_detections": len(detections),
            "cropped_images": cropped_image_paths,
            "detections": detection_results,
            "save_directory": save_directory,
            "results_file": results_file
        }, ensure_ascii=False, indent=2)
        
    except Exception as e:
        return json.dumps({
            "error": f"Error during detection and cropping: {str(e)}",
            "cropped_images": []
        })

if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport='stdio') 