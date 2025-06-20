#!/usr/bin/env python3
"""
run_visual_article_generator.py
Author: Ying-Cong Chen (yingcong.ian.chen@gmail.com)
Date: 2025-01-14
Description: Simple launcher for VisualArticleTool
License: MIT License
"""

import os
import sys

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from tools.visual_article.visual_article_agent import VisualArticleTool

if __name__ == "__main__":
    VisualArticleTool.main() 