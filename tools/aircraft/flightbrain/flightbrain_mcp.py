import os
import asyncio
from typing import Dict, Any, Optional
from datetime import datetime
from mcp.server.fastmcp import FastMCP
from openai import OpenAI
from dotenv import load_dotenv
from PIL import Image
import base64
import io
import json

# Load environment variables
load_dotenv()

# Initialize FastMCP server
mcp = FastMCP("flightbrain")

class AircraftState:
    """Represents the current state of the aircraft"""
    def __init__(self):
        self.position = {"x": 0.0, "y": 0.0, "z": 100.0}  # meters
        self.velocity = {"x": 0.0, "y": 0.0, "z": 0.0}    # m/s
        self.attitude = {"pitch": 0.0, "roll": 0.0, "yaw": 0.0}  # degrees
        self.status = "hovering"
        self.altitude = 100.0  # meters
        self.last_update = datetime.now()
        self.safety_status = "unknown"  # Can be: unknown, safe, caution, unsafe

    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary"""
        return {
            "position": self.position,
            "velocity": self.velocity,
            "attitude": self.attitude,
            "status": self.status,
            "altitude": self.altitude,
            "last_update": self.last_update.isoformat(),
            "safety_status": self.safety_status
        }

# Global aircraft state
aircraft = AircraftState()

def normalize_path(path: str) -> str:
    """Normalize a file path by expanding ~ to user's home directory and resolving relative paths."""
    expanded_path = os.path.expanduser(path)
    if not os.path.isabs(expanded_path):
        expanded_path = os.path.abspath(expanded_path)
    return expanded_path

def encode_image(image: Image.Image, size: tuple[int, int] = (512, 512)) -> str:
    """Encode and resize image to base64 string."""
    image.thumbnail(size)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

def load_image(image_path: str, size_limit: tuple[int, int] = (512, 512)) -> tuple[str, dict]:
    """Load and prepare image for model input."""
    meta_info = {}
    image = Image.open(image_path)
    meta_info['width'], meta_info['height'] = image.size
    base64_image = encode_image(image, size_limit)
    return base64_image, meta_info

@mcp.tool()
async def hover(duration: float = 0.0) -> str:
    """
    Make the aircraft hover at current position.
    
    Args:
        duration (float): Duration to hover in seconds (optional)
    
    Returns:
        str: JSON string containing hover command status and aircraft state
    """
    global aircraft
    
    aircraft.status = "hovering"
    aircraft.velocity = {"x": 0.0, "y": 0.0, "z": 0.0}
    aircraft.last_update = datetime.now()
    
    print("The aircraft is hovering")
    if duration > 0:
        print(f"Hovering for {duration} seconds")
    
    return json.dumps({
        "status": "success",
        "message": f"Aircraft hovering{f' for {duration} seconds' if duration > 0 else ''}",
        "aircraft_state": aircraft.to_dict()
    }, indent=2)

@mcp.tool()
async def move_forward(distance: float = 1.0) -> str:
    """
    Move the aircraft forward.
    
    Args:
        distance (float): Distance to move forward in meters
    
    Returns:
        str: JSON string containing forward movement status and aircraft state
    """
    global aircraft
    
    aircraft.position["x"] += distance
    aircraft.status = "moving_forward"
    aircraft.last_update = datetime.now()
    
    print(f"The aircraft is moving forward {distance} meters")
    
    return json.dumps({
        "status": "success",
        "message": f"Aircraft moving forward {distance} meters",
        "aircraft_state": aircraft.to_dict()
    }, indent=2)

@mcp.tool()
async def move_backward(distance: float = 1.0) -> str:
    """
    Move the aircraft backward.
    
    Args:
        distance (float): Distance to move backward in meters
    
    Returns:
        str: JSON string containing backward movement status and aircraft state
    """
    global aircraft
    
    aircraft.position["x"] -= distance
    aircraft.status = "moving_backward"
    aircraft.last_update = datetime.now()
    
    print(f"The aircraft is moving backward {distance} meters")
    
    return json.dumps({
        "status": "success",
        "message": f"Aircraft moving backward {distance} meters",
        "aircraft_state": aircraft.to_dict()
    }, indent=2)

@mcp.tool()
async def move_left(distance: float = 1.0) -> str:
    """
    Move the aircraft left.
    
    Args:
        distance (float): Distance to move left in meters
    
    Returns:
        str: JSON string containing left movement status and aircraft state
    """
    global aircraft
    
    aircraft.position["y"] -= distance
    aircraft.status = "moving_left"
    aircraft.last_update = datetime.now()
    
    print(f"The aircraft is moving left {distance} meters")
    
    return json.dumps({
        "status": "success",
        "message": f"Aircraft moving left {distance} meters",
        "aircraft_state": aircraft.to_dict()
    }, indent=2)

@mcp.tool()
async def move_right(distance: float = 1.0) -> str:
    """
    Move the aircraft right.
    
    Args:
        distance (float): Distance to move right in meters
    
    Returns:
        str: JSON string containing right movement status and aircraft state
    """
    global aircraft
    
    aircraft.position["y"] += distance
    aircraft.status = "moving_right"
    aircraft.last_update = datetime.now()
    
    print(f"The aircraft is moving right {distance} meters")
    
    return json.dumps({
        "status": "success",
        "message": f"Aircraft moving right {distance} meters",
        "aircraft_state": aircraft.to_dict()
    }, indent=2)

@mcp.tool()
async def ascend(distance: float = 1.0) -> str:
    """
    Make the aircraft ascend (gain altitude).
    
    Args:
        distance (float): Distance to ascend in meters
    
    Returns:
        str: JSON string containing ascend status and aircraft state
    """
    global aircraft
    
    aircraft.altitude += distance
    aircraft.position["z"] += distance
    aircraft.status = "ascending"
    aircraft.last_update = datetime.now()
    
    print(f"The aircraft is ascending {distance} meters")
    
    return json.dumps({
        "status": "success",
        "message": f"Aircraft ascending {distance} meters",
        "aircraft_state": aircraft.to_dict()
    }, indent=2)

@mcp.tool()
async def descend(distance: float = 1.0) -> str:
    """
    Make the aircraft descend (lose altitude).
    
    Args:
        distance (float): Distance to descend in meters
    
    Returns:
        str: JSON string containing descend status and aircraft state
    """
    global aircraft
    
    if aircraft.altitude - distance >= 0:
        aircraft.altitude -= distance
        aircraft.position["z"] -= distance
        aircraft.status = "descending"
        aircraft.last_update = datetime.now()
        
        print(f"The aircraft is descending {distance} meters")
        
        return json.dumps({
            "status": "success",
            "message": f"Aircraft descending {distance} meters",
            "aircraft_state": aircraft.to_dict()
        }, indent=2)
    else:
        print("Cannot descend below ground level")
        return json.dumps({
            "status": "error",
            "message": "Cannot descend below ground level",
            "aircraft_state": aircraft.to_dict()
        }, indent=2)

@mcp.tool()
async def rotate(angle: float = 0.0) -> str:
    """
    Rotate the aircraft by specified degrees.
    
    Args:
        angle (float): Angle to rotate in degrees
    
    Returns:
        str: JSON string containing rotation status and aircraft state
    """
    global aircraft
    
    aircraft.attitude["yaw"] = (aircraft.attitude["yaw"] + angle) % 360
    aircraft.status = "rotating"
    aircraft.last_update = datetime.now()
    
    print(f"The aircraft is rotating {angle} degrees")
    
    return json.dumps({
        "status": "success",
        "message": f"Aircraft rotating {angle} degrees",
        "aircraft_state": aircraft.to_dict()
    }, indent=2)

@mcp.tool()
async def hover_turn(angle: float = 0.0) -> str:
    """
    Perform a hover turn - rotate the aircraft while maintaining position and hovering.
    
    Args:
        angle (float): Angle to turn in degrees (positive for clockwise, negative for counter-clockwise)
    
    Returns:
        str: JSON string containing hover turn status and aircraft state
    """
    global aircraft
    
    # Ensure aircraft is hovering
    aircraft.status = "hover_turning"
    aircraft.velocity = {"x": 0.0, "y": 0.0, "z": 0.0}  # Maintain position
    
    # Apply rotation
    aircraft.attitude["yaw"] = (aircraft.attitude["yaw"] + angle) % 360
    aircraft.last_update = datetime.now()
    
    direction = "clockwise" if angle >= 0 else "counter-clockwise"
    print(f"The aircraft is performing a hover turn {abs(angle)} degrees {direction}")
    
    return json.dumps({
        "status": "success",
        "message": f"Aircraft performing hover turn {abs(angle)} degrees {direction}",
        "aircraft_state": aircraft.to_dict()
    }, indent=2)

@mcp.tool()
async def land() -> str:
    """
    Begin landing sequence.
    
    Returns:
        str: JSON string containing landing status and aircraft state
    """
    global aircraft
    
    if aircraft.altitude > 0:
        aircraft.altitude = 0
        aircraft.position["z"] = 0
        aircraft.status = "landed"
        aircraft.last_update = datetime.now()
        
        print("The aircraft has landed")
        
        return json.dumps({
            "status": "success",
            "message": "Aircraft has landed",
            "aircraft_state": aircraft.to_dict()
        }, indent=2)
    else:
        print("The aircraft is already on the ground")
        return json.dumps({
            "status": "info",
            "message": "Aircraft is already on the ground",
            "aircraft_state": aircraft.to_dict()
        }, indent=2)

@mcp.tool()
async def get_aircraft_state() -> str:
    """
    Get the current state of the aircraft.
    
    Returns:
        str: JSON string containing current aircraft state
    """
    return json.dumps({
        "status": "success",
        "aircraft_state": aircraft.to_dict()
    }, indent=2)

@mcp.tool()
async def analyze_flight_situation(image_path: str) -> str:
    """
    Analyze the flight situation using visual input.
    
    Args:
        image_path (str): Path to the image showing aircraft and landing area
        
    Returns:
        str: Detailed analysis of the flight situation
    """
    image_path = normalize_path(image_path)
    base64_image, meta_info = load_image(image_path, (512, 512))

    client = OpenAI(
        api_key=os.getenv('QWEN_API_KEY'),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    
    prompt = """请分析图中飞行器与降落区域的相对位置关系，并详细描述：
1. 飞行器与停机坪的相对距离（近、中、远）和方位（正上方、前方、侧方等）
2. 下降路径上是否存在障碍物（建筑物、树木、电线等）
3. 停机坪周围的环境状况（障碍物分布、空间开阔程度）
4. 飞行器当前的离地高度（过高、适中、过低）
5. 建议的下一步操作（继续下降、调整位置、保持悬停等）

请以结构化的方式输出分析结果。"""

    completion = client.chat.completions.create(
        model="qwen-vl-max",
        messages=[{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f'data:image/png;base64,{base64_image}'}}
        ]}]
    )
    
    # Update aircraft safety status based on analysis
    response = completion.choices[0].message.content
    
    # Simple heuristic to update safety status based on keywords in the response
    if any(word in response.lower() for word in ["危险", "障碍", "unsafe", "danger"]):
        aircraft.safety_status = "unsafe"
    elif any(word in response.lower() for word in ["注意", "caution", "warning"]):
        aircraft.safety_status = "caution"
    else:
        aircraft.safety_status = "safe"
    
    return response

@mcp.tool()
async def read_latest_safety_result() -> str:
    """
    Read the latest safety analysis result from file.
    
    Returns:
        str: JSON string containing the latest safety analysis result
    """
    try:
        latest_file = "./safety_check/latest_safety_result.json"
        latest_file = normalize_path(latest_file)
        
        if os.path.exists(latest_file):
            with open(latest_file, 'r', encoding='utf-8') as f:
                result_data = json.load(f)
            return json.dumps(result_data, ensure_ascii=False, indent=2)
        else:
            return json.dumps({
                "status": "no_result",
                "message": "No safety analysis result found. Please run safety analysis first."
            })
    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": f"Error reading safety result: {str(e)}"
        })

if __name__ == "__main__":
    mcp.run(transport='stdio') 