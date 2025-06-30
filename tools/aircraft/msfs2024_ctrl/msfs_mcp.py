import requests
import json
import math
import time

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

# API端点
API_URL = "http://10.4.147.50:5000/set"


def set_flight_parameter(param_name, param_value):
    """
    设置飞行参数
    
    Args:
        param_name (str): 参数名称
        param_value: 参数值
        
    Returns:
        str: 返回消息
    """
    # 准备请求数据
    payload = {
        "name": param_name,
        "val": param_value
    }

    try:
        # 发送PUT请求到API
        response = requests.put(API_URL, json=payload)
        
        # 检查响应状态码
        if response.status_code == 200:
            # 解析JSON响应
            data = response.json()
            return f"设置成功: {data['message']}"
        else:
            if response.text:
                try:
                    error_data = response.json()
                    return f"请求失败，状态码: {response.status_code}, 错误信息: {error_data['message']}"
                except:
                    return f"请求失败，状态码: {response.status_code}, 响应内容: {response.text}"
            else:
                return f"请求失败，状态码: {response.status_code}"
            
    except requests.exceptions.RequestException as e:
        return f"请求错误: {e}"
    except json.JSONDecodeError:
        return "无法解析JSON响应"
    except Exception as e:
        return f"发生错误: {e}"

@mcp.tool()
def move_forward(time_s):
    """
    Make the aircraft move forward
    
    Args:
        duration (float): Duration to move forward in seconds
    
    Returns:
        None
    """
    print("Moving forward")
    hover()
    set_flight_parameter("GENERAL_ENG_THROTTLE_LEVER_POSITION_1", 99)
    time.sleep(time_s)
    hover()
    print("Moving forward done")
    pass

@mcp.tool()
def move_backward(time_s):
    """
    Make the aircraft move backward
    
    Args:
        duration (float): Duration to move backward in seconds
    
    Returns:
        None
    """
    hover()
    set_flight_parameter("GENERAL_ENG_THROTTLE_LEVER_POSITION_1", 0)
    time.sleep(time_s)
    hover()
    print("Moving backward done")
    pass

@mcp.tool()
def move_left(time_s):
    """
    Make the aircraft move left
    
    Args:
        duration (float): Duration to move left in seconds
    
    Returns:
        None
    """
    print("Moving left")
    hover()
    set_flight_parameter("AILERON_POSITION", -1.0)
    time.sleep(time_s)
    hover()
    print("Moving left done")
    pass

@mcp.tool()
def move_right(time_s):
    """
    Make the aircraft move right
    
    Args:
        duration (float): Duration to move right in seconds
    
    Returns:
        None
    """
    print("Moving right")
    hover()
    set_flight_parameter("AILERON_POSITION", 1.0)
    time.sleep(time_s)
    hover()
    print("Moving right done")
    pass

@mcp.tool()
def move_up(time_s):
    """
    Make the aircraft move ascend
    
    Args:
        duration (float): Duration to move up in seconds
    
    """
    print("Moving up")
    hover()
    set_flight_parameter("ELEVATOR_POSITION", 1.0)
    time.sleep(time_s)
    hover()
    print("Moving up done")
    pass

@mcp.tool()
def move_down(time_s):
    """
    Make the aircraft move descend
    
    Args:
        duration (float): Duration to move down in seconds
    
    """
    print("Moving down")
    hover()
    set_flight_parameter("ELEVATOR_POSITION", 0.5)
    time.sleep(time_s)
    hover()
    print("Moving down done")
    pass

@mcp.tool()
def hover():
    """
    Make the aircraft hover
    
    Args:
        None
    
    """
    print("Hovering")
    set_flight_parameter("GENERAL_ENG_THROTTLE_LEVER_POSITION_1", 50)
    set_flight_parameter("RUDDER_POSITION", 0.0)
    set_flight_parameter("AILERON_POSITION", 0.0)
    set_flight_parameter("ELEVATOR_POSITION", 0.0)
    time.sleep(0.1)
    pass

@mcp.tool()
def hover_turn_left(time_s):
    """
    Make the aircraft hover and turn left
    
    Args:
        duration (float): Duration to turn left in seconds
    
    """
    print("Hover turning left")
    hover()
    set_flight_parameter("RUDDER_POSITION", -0.25)
    time.sleep(time_s)
    hover()
    print("Hover turning left done")
    pass

@mcp.tool() 
def hover_turn_right(time_s):
    """
    Make the aircraft hover and turn right
    
    Args:
        duration (float): Duration to turn right in seconds
    
    """
    print("Hover turning right")
    hover()
    set_flight_parameter("RUDDER_POSITION", 0.25)
    time.sleep(time_s)
    hover()
    print("Hover turning right done")
    pass

if __name__ == "__main__":
    hover()
    hover_turn_right(10)