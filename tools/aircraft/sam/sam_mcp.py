# 此文件保存SAM工具！
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
import numpy as np
from PIL import Image
import os
import glob
import time
import threading
from .sam_utils import SAM_tool, SAMClient
import gradio as gr
import asyncio
from scipy.ndimage import label

# helper functions
def create_loading_image(original_img):
    # 临时用一下
    if original_img is None:
        return None
    
    img_array = original_img.copy()
    h, w = img_array.shape[:2]
    
    # 创建半透明遮罩
    overlay = np.zeros_like(img_array)
    overlay[:, :] = [128, 128, 128]  # 灰色遮罩
    
    # 应用半透明效果
    alpha = 0.7
    img_array = (img_array * (1 - alpha) + overlay * alpha).astype(np.uint8)
    
    return img_array
def apply_green_overlay(img_array, mask, alpha=0.3):
    # 创建绿色遮罩
    green_overlay = np.zeros_like(img_array)
    green_overlay[:, :, 1] = 255  # 绿色通道
    
    # 将mask扩展到3个通道
    mask_3d = np.stack([mask, mask, mask], axis=2)
    
    # 确保mask和图片尺寸匹配
    if mask_3d.shape[:2] != img_array.shape[:2]:
        print(f"警告: mask尺寸 {mask_3d.shape[:2]} 与图片尺寸 {img_array.shape[:2]} 不匹配")
        return img_array, mask_3d
    
    # 应用半透明绿色遮罩
    overlay_region = mask_3d != 0
    img_array[overlay_region] = (
        img_array[overlay_region] * (1 - alpha) + 
        green_overlay[overlay_region] * alpha
    ).astype(np.uint8)
    return img_array, mask_3d
def draw_bounding_box(image, overlay_region, thickness=10, color=[255, 0, 0]):
    bbox_img = image.copy()
    # 找到mask的边界框
    mask_coords = np.where(overlay_region[:, :, 0])  # 使用第一个通道即可
    if len(mask_coords[0]) > 0:  # 确保有mask区域
        min_y, max_y = np.min(mask_coords[0]), np.max(mask_coords[0])
        min_x, max_x = np.min(mask_coords[1]), np.max(mask_coords[1])
        # 绘制边界框
        # 绘制水平线
        for t in range(thickness):
            # 上边
            if min_y + t < bbox_img.shape[0]:
                bbox_img[min_y + t, min_x:max_x + 1] = color
            # 下边
            if max_y - t >= 0:
                bbox_img[max_y - t, min_x:max_x + 1] = color
        
        # 绘制垂直线
        for t in range(thickness):
            # 左边
            if min_x + t < bbox_img.shape[1]:
                bbox_img[min_y:max_y + 1, min_x + t] = color
            # 右边
            if max_x - t >= 0:
                bbox_img[min_y:max_y + 1, max_x - t] = color
    return bbox_img
def draw_mask_boundary(image, mask, color=[255, 0, 0], thickness=2):
    if mask is None or mask.size == 0:
        return image
    
    img_array = image.copy()
    h, w = mask.shape
    
    # 创建边缘检测mask
    boundary_mask = np.zeros_like(mask, dtype=bool)
    
    # 检测边缘：如果当前像素是mask区域，但周围有非mask区域，则为边缘
    for y in range(h):
        for x in range(w):
            if mask[y, x] != 0:  # 当前像素在mask内（修改：使用!= 0而不是== 1）
                # 检查8邻域
                is_boundary = False
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dy == 0 and dx == 0:
                            continue
                        ny, nx = y + dy, x + dx
                        # 如果邻域超出边界或者邻域不在mask内，则当前像素是边缘
                        if (ny < 0 or ny >= h or nx < 0 or nx >= w or 
                            mask[ny, nx] == 0):  # 这里保持== 0，检查邻域是否为0
                            is_boundary = True
                            break
                    if is_boundary:
                        break
                
                if is_boundary:
                    boundary_mask[y, x] = True
    
    # 根据thickness参数加粗边缘
    if thickness > 1:
        thick_boundary = np.zeros_like(boundary_mask)
        boundary_coords = np.where(boundary_mask)
        for y, x in zip(boundary_coords[0], boundary_coords[1]):
            for dy in range(-thickness//2, thickness//2 + 1):
                for dx in range(-thickness//2, thickness//2 + 1):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w:
                        thick_boundary[ny, nx] = True
        boundary_mask = thick_boundary
    
    # 将边缘应用到图像上
    boundary_coords = np.where(boundary_mask)
    if len(boundary_coords[0]) > 0:
        img_array[boundary_coords[0], boundary_coords[1]] = color
    return img_array
def find_point_connected_component(mask, point_x, point_y):
    """
    找到mask中包含指定点的联通区域，将其他联通区域置为0
    
    Args:
        mask: 输入的mask数组
        point_x: 点击点的x坐标
        point_y: 点击点的y坐标
        
    Returns:
        cleaned_mask: 只保留包含指定点的联通区域的mask
    """
    if mask is None or mask.size == 0:
        return mask
    
    # 创建二值化mask（非0的地方为True）
    binary_mask = mask != 0
    
    # 找到所有联通区域
    labeled_mask, num_features = label(binary_mask)
    
    if num_features == 0:
        print(">>> 没有找到任何联通区域")
        return mask
    
    # 检查点击点是否在图像范围内
    h, w = mask.shape
    if not (0 <= point_x < w and 0 <= point_y < h):
        print(f">>> 点击点({point_x}, {point_y})超出图像范围({w}, {h})")
        return mask
    
    # 获取点击点所在的联通区域标签
    target_label = labeled_mask[point_y, point_x]
    
    if target_label == 0:
        print(f">>> 点击点({point_x}, {point_y})不在任何mask区域内")
        return mask
    
    # 计算目标联通区域的大小
    target_size = np.sum(labeled_mask == target_label)
    
    print(f">>> 找到{num_features}个联通区域，目标区域大小: {target_size}")
    
    # 创建新的mask，只保留包含点击点的联通区域
    cleaned_mask = np.zeros_like(mask)
    target_component_coords = labeled_mask == target_label
    cleaned_mask[target_component_coords] = mask[target_component_coords]
    
    return cleaned_mask
def center_crop_mask_region(image, mask, crop_size):
    """
    围绕mask非0值的中心进行center crop并保存
    
    Args:
        image: 要裁剪的图像数组
        mask: mask数组，用于确定中心位置
        crop_size: 裁剪后的尺寸，默认512x512
        save_path: 保存路径
        
    Returns:
        cropped_image: 裁剪后的图像，如果没有mask区域则返回None
    """
    # 找到mask中非0值的中心
    mask_coords = np.where(mask != 0)
    if len(mask_coords[0]) == 0:
        print('>>> 没有找到mask区域，无法进行center crop')
        return None
    
    # 计算mask中心
    center_y = int(np.mean(mask_coords[0]))
    center_x = int(np.mean(mask_coords[1]))
    
    # 计算mask的实际尺寸
    min_y, max_y = np.min(mask_coords[0]), np.max(mask_coords[0])
    min_x, max_x = np.min(mask_coords[1]), np.max(mask_coords[1])
    mask_height = max_y - min_y + 1
    mask_width = max_x - min_x + 1
    
    # 获取原图尺寸
    h, w = image.shape[:2]
    
    # 分别计算x和y维度的实际crop尺寸
    # 处理x维度(width)
    if mask_width > crop_size:
        expanded_width = int(mask_width * 1.2)
        actual_crop_width = min(expanded_width, w)
        print(f'>>> mask宽度({mask_width})大于crop_size({crop_size})，x维度扩展到{actual_crop_width}')
    else:
        actual_crop_width = min(crop_size, w)
    
    # 处理y维度(height)
    if mask_height > crop_size:
        expanded_height = int(mask_height * 1.2)
        actual_crop_height = min(expanded_height, h)
        print(f'>>> mask高度({mask_height})大于crop_size({crop_size})，y维度扩展到{actual_crop_height}')
    else:
        actual_crop_height = min(crop_size, h)
    
    half_width = actual_crop_width // 2
    half_height = actual_crop_height // 2
    
    # 计算crop区域，确保不超出图像边界
    start_x = max(0, min(center_x - half_width, w - actual_crop_width))
    end_x = min(w, start_x + actual_crop_width)
    start_y = max(0, min(center_y - half_height, h - actual_crop_height))
    end_y = min(h, start_y + actual_crop_height)
    
    # 如果图像尺寸小于crop尺寸，调整到图像边界
    if w < actual_crop_width:
        start_x = 0
        end_x = w
        actual_crop_width = w
    
    if h < actual_crop_height:
        start_y = 0
        end_y = h
        actual_crop_height = h
    
    # 检查裁剪区域有效性
    if start_x >= end_x or start_y >= end_y:
        print(f'>>> 裁剪区域无效: start_x={start_x}, end_x={end_x}, start_y={start_y}, end_y={end_y}')
    if start_x < 0 or start_y < 0 or end_x > w or end_y > h:
        print(f'>>> 裁剪区域超出边界: start_x={start_x}, end_x={end_x}, start_y={start_y}, end_y={end_y}')
    

    # 执行crop
    cropped_image = image[start_y:end_y, start_x:end_x]
    # Image.fromarray(cropped_image).save("./sam/tmp/cropped_image_before_padding.png")
    # 如果裁剪后的尺寸不足目标尺寸，进行padding
    if cropped_image.shape[0] < actual_crop_height or cropped_image.shape[1] < actual_crop_width:
        # 创建目标尺寸的空白图像
        padded_img = np.zeros((actual_crop_height, actual_crop_width, 3), dtype=np.uint8)
        # 计算padding位置
        pad_y = (actual_crop_height - cropped_image.shape[0]) // 2
        pad_x = (actual_crop_width - cropped_image.shape[1]) // 2
        padded_img[pad_y:pad_y+cropped_image.shape[0], 
                  pad_x:pad_x+cropped_image.shape[1]] = cropped_image
        cropped_image = padded_img
    
    return cropped_image

# core
class SAM_TOOL:
    def __init__(self, mask_type="boundary", crop_size=512):
        load_dotenv()
        # MCP server
        mcp = FastMCP("SAM")
        #mcp.tool(self.detect)
        # SAM Client
        self.client = SAMClient(server_url="http://10.30.58.120:5000")
        # 遮罩方法
        self.mask_type = mask_type # boundary, mask, bbox
        # 返回大小
        self.crop_size = crop_size
        # 当前图片
        self.img = None

    # 加载单张图片 - 可无损替换为加载视频帧
    def load_frame(self, img_pth):
        # 检测文件存在
        if not os.path.exists(img_pth):
            return None, f"图片文件不存在: {img_pth}"
        # 加载
        img = Image.open(img_pth)
        # 处理不同格式的图片
        if img.mode == 'RGBA':
            # 将RGBA转换为RGB
            background = Image.new('RGB', img.size, (255, 255, 255))  # 白色背景
            background.paste(img, mask=img.split()[-1])  # 使用alpha通道作为mask
            img = background
        elif img.mode not in ['RGB', 'L']:
            # 其他格式转换为RGB
            img = img.convert('RGB')
        # 保存到np array
        img = np.array(img)
        self.img = img
        return img
    
    # 图片+遮罩 = 目标图片
    def apply_mask_overlay(self, image, mask, point_x, point_y, alpha=0.3):
        if mask is None:
            return image
        img_array = image.copy()
        # 检查图片通道数并处理
        if len(img_array.shape) == 3:
            if img_array.shape[2] == 4:  # RGBA格式
                # 转换为RGB格式
                img_array = img_array[:, :, :3]
            elif img_array.shape[2] != 3:  # 其他格式
                return image  # 不支持的格式，返回原图
        else:
            return image  # 不是3维数组，返回原图
        
        # 找到点击点所在的联通区域并清理mask
        cleaned_mask = find_point_connected_component(mask, point_x, point_y)
        
        # 应用绿色遮罩
        img_array, mask_3d = apply_green_overlay(img_array, cleaned_mask, alpha)
        # 绘制红色边缘轮廓
        boundary_img = draw_mask_boundary(image, cleaned_mask, color=[255, 0, 0], thickness=10)

        return img_array, boundary_img, center_crop_mask_region(boundary_img, cleaned_mask, crop_size=self.crop_size)

    # 核心检测 - 注意异步！
    async def detect(self, image, points:gr.SelectData):
        image = self.img
        img_array = image.copy()
        image = Image.fromarray(image)
        # 获取点击的坐标 - 支持点击或普通[x,y]
        if isinstance(points, gr.SelectData):
            x, y = points.index[0], points.index[1]
        else:
            x, y = points
        loading_image = create_loading_image(img_array)

        #yield loading_image, f"正在识别...\n当前识别坐标: ({x}, {y})"
        #await asyncio.sleep(1)
        # 调用SAM模型
        #yield loading_image, f"正在调用SAM模型进行分割...\n当前点击坐标: ({x}, {y})"
        #await asyncio.sleep(1)
        mask = SAM_tool(self.client, image, [[x, y]])
        
        # 处理图像
        #yield loading_image, f"正在应用mask叠加...\n当前点击坐标: ({x}, {y})"
        #await asyncio.sleep(1)
        
        img_array, boundary, boundary_cropped = self.apply_mask_overlay(img_array, mask, x, y)
        
        # save cropped_image to ./sam/tmp/cropped_image.png
        Image.fromarray(boundary_cropped).save("./sam/tmp/cropped_image.png")
        # save boundary_img to ./sam/tmp/boundary_img.png
        Image.fromarray(boundary).save("./sam/tmp/boundary_img.png")

        # 绘制标记点
        #yield loading_image, f"正在绘制标记点...\n当前点击坐标: ({x}, {y})"
        #h, w = img_array.shape[:2]
        #if 0 <= x < w and 0 <= y < h:
        #    radius = 8
        #    for dx in range(-radius, radius + 1):
        #        for dy in range(-radius, radius + 1):
        #            nx, ny = x + dx, y + dy
        #            if 0 <= nx < w and 0 <= ny < h and dx*dx + dy*dy <= radius*radius:
        #                img_array[ny, nx] = [255, 0, 0]  # 红色标记
        
        # 返回最终结果
        # self.masked_img = img_array
        return img_array, boundary_cropped, f"处理完成\n当前点击坐标: ({x}, {y})"

    #async def detect_tool(self, image, points:gr.SelectData):
    #    _, boundary, _ = self.detect(image, points)
    #    return boundary
