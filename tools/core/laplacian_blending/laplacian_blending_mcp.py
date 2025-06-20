import cv2
import numpy as np
import urllib.request
import os
from uuid import uuid4
from urllib.parse import urlparse

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("laplacian_blending")

def load_image(path_or_url):
    """从 URL 或本地路径加载图像"""
    if urlparse(path_or_url).scheme in ('http', 'https'):
        try:
            resp = urllib.request.urlopen(path_or_url)
            image_data = np.asarray(bytearray(resp.read()), dtype=np.uint8)
            image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
        except Exception as e:
            raise ValueError(f"下载图像失败: {e}")
    else:
        if not os.path.exists(path_or_url):
            raise FileNotFoundError(f"文件不存在: {path_or_url}")
        image = cv2.imread(path_or_url)
    if image is None:
        raise ValueError(f"无法读取图像: {path_or_url}")
    return image

@mcp.tool()
def laplacian_blending(path_A, path_B, path_mask, output_dir="output", levels=6):
    """Blend two images using Laplacian pyramid blending with a mask.
    
    Args:
        image_a_path: Path/URL to the first input image
        image_b_path: Path/URL to the second input image
        mask_path: Path/URL to the mask image (white=use image A, black=use image B)
        output_dir: Directory to save the blended result (default: "output")
        pyramid_levels: Number of pyramid levels for blending (default: 6)
        
    Returns:
        str: Path to the saved blended image
        
    Raises:
        ValueError: If any input image is invalid or cannot be loaded
    """
    # 加载图像
    A = load_image(path_A)
    B = load_image(path_B)
    M = load_image(path_mask)

    # 尺寸对齐
    min_shape = (min(A.shape[0], B.shape[0]), min(A.shape[1], B.shape[1]))
    A = cv2.resize(A, (min_shape[1], min_shape[0]))
    B = cv2.resize(B, (min_shape[1], min_shape[0]))
    M = cv2.resize(M, (min_shape[1], min_shape[0]))

    # 处理 mask
    if len(M.shape) == 3:
        M = cv2.cvtColor(M, cv2.COLOR_BGR2GRAY)
    M = M.astype(np.float32) / 255.0
    M = cv2.merge([M, M, M])

    # 构建金字塔
    def build_pyramid(img, levels):
        gp = [img.astype(np.float32)]
        for _ in range(levels):
            img = cv2.pyrDown(img)
            gp.append(img.astype(np.float32))
        return gp

    def build_laplacian_pyramid(gp):
        lp = []
        for i in range(len(gp) - 1):
            GE = cv2.pyrUp(gp[i + 1], dstsize=(gp[i].shape[1], gp[i].shape[0]))
            L = cv2.subtract(gp[i], GE)
            lp.append(L)
        lp.append(gp[-1])
        return lp

    gpA = build_pyramid(A, levels)
    gpB = build_pyramid(B, levels)
    gpM = build_pyramid(M, levels)
    lpA = build_laplacian_pyramid(gpA)
    lpB = build_laplacian_pyramid(gpB)

    # 融合
    LS = [gm * la + (1.0 - gm) * lb for la, lb, gm in zip(lpA, lpB, gpM)]

    # 重构图像
    result = LS[-1]
    for i in range(levels - 1, -1, -1):
        result = cv2.pyrUp(result, dstsize=(LS[i].shape[1], LS[i].shape[0]))
        result = cv2.add(result, LS[i])
    result = np.clip(result, 0, 255).astype(np.uint8)

    # 保存结果
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    filename = f"blended_{uuid4().hex[:8]}.png"
    output_path = os.path.join(output_dir, filename)
    cv2.imwrite(output_path, result)

    return output_path

if __name__ == "__main__":
    mcp.run(transport='stdio') 