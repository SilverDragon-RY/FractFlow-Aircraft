#!/usr/bin/env python3
"""
run_weather.py
Author: Ying-Cong Chen (yingcong.ian.chen@gmail.com)
Date: 2025-01-14
Description: Simple launcher for WeatherTool
License: MIT License
"""

import os
import sys

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from tools.weather.weather_agent import WeatherAgent

if __name__ == "__main__":
    WeatherAgent.main() 