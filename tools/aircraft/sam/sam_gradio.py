import gradio as gr
import numpy as np
from PIL import Image
import os
import glob
import time
import threading

from sam_utils import SAM_tool, SAMClient

# 配置固定的图片路径
# 请根据实际情况修改路径
IMAGE_PATH = "./tmp/camera/test.jpg"
MASK_DIR = "./tmp/individual_masks"  # mask文件夹路径
RELOAD_TRIGGER_FILE = "./tmp/reload_trigger.txt"  # 重新加载触发文件

# 存储最新点击坐标、原始图片和masks
latest_click_point = None
original_image = None
masks = []  # 存储所有mask数据
mask_files = []  # 存储mask文件名

# Gradio组件的全局引用
image_display_ref = None
coordinate_info_ref = None
client = None

def load_masks():
    """加载所有mask文件"""
    global masks, mask_files
    
    masks = []
    mask_files = []
    
    # 查找所有individual_masks_*.png文件
    mask_pattern = os.path.join(MASK_DIR, "individual_masks_*.png")
    mask_paths = glob.glob(mask_pattern)
    mask_paths.sort()  # 按文件名排序
    
    for mask_path in mask_paths:
        try:
            # 加载mask图像
            mask_img = Image.open(mask_path).convert('L')  # 转换为灰度
            mask_array = np.array(mask_img)
            
            # 将mask二值化（假设白色为1，黑色为0）
            mask_binary = (mask_array > 128).astype(np.uint8)
            
            masks.append(mask_binary)
            mask_files.append(os.path.basename(mask_path))
            
        except Exception as e:
            print(f"无法加载mask文件 {mask_path}: {str(e)}")
    
    return len(masks)

def find_mask_at_position(x, y):
    """根据点击位置找到对应的mask"""
    global masks
    
    for i, mask in enumerate(masks):
        h, w = mask.shape
        if 0 <= x < w and 0 <= y < h:
            if mask[y, x] == 1:  # 注意：numpy数组是[行,列]即[y,x]
                return i, mask
    
    return None, None

def apply_green_overlay(img_array, mask, alpha=0.3):
    """应用绿色半透明遮罩到图像上"""
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
    print(np.unique(mask_3d))
    overlay_region = mask_3d != 0
    img_array[overlay_region] = (
        img_array[overlay_region] * (1 - alpha) + 
        green_overlay[overlay_region] * alpha
    ).astype(np.uint8)
    print('>>> overlay finished')
    
    return img_array, mask_3d

def draw_bounding_box(image, overlay_region, thickness=10, color=[255, 0, 0]):
    """在mask区域绘制边界框"""
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
        
        print(f'>>> bounding box: ({min_x}, {min_y}) to ({max_x}, {max_y})')
    
    return bbox_img

def draw_mask_boundary(image, mask, color=[255, 0, 0], thickness=2):
    """根据mask绘制红色的边缘轮廓"""
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
    
    print(f'>>> boundary drawn with {len(boundary_coords[0])} pixels')
    
    return img_array

def apply_mask_overlay(image, mask, alpha=0.3):
    print('>>> apply_mask_overlay: ', mask.shape)
    """将mask以绿色半透明方式叠加到图像上"""
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
    
    # 应用绿色遮罩
    img_array, mask_3d = apply_green_overlay(img_array, mask, alpha)
    
    # 保存overlay结果和mask
    Image.fromarray(img_array).save('./tmp/test_overlay.png')
    Image.fromarray(mask_3d).save('./tmp/test_mask.png')
    
    # 绘制红色边界框并保存
    overlay_region = mask_3d != 0
    bbox_img = draw_bounding_box(image, overlay_region, thickness=10, color=[255, 0, 0])
    Image.fromarray(bbox_img).save('./tmp/test_bbox.png')
    
    # 绘制红色边缘轮廓并保存
    boundary_img = draw_mask_boundary(image, mask, color=[255, 0, 0], thickness=10)
    Image.fromarray(boundary_img).save('./tmp/test_boundary.png')
    
    return img_array

def create_loading_image(original_img, message="正在处理..."):
    """创建带有加载提示的图片"""
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

def handle_image_click(image, evt: gr.SelectData):
    """处理图片点击事件 - 使用generator实现实时更新"""
    global latest_click_point, original_image
    
    # 获取点击的坐标
    x, y = evt.index[0], evt.index[1]
    latest_click_point = (x, y)
    
    print('>>> click point: ', x, y)
    
    # 立即显示加载状态
    if original_image is not None:
        loading_img = create_loading_image(original_image, "正在处理点击事件...")
        yield loading_img, f"正在处理点击事件...\n当前点击坐标: ({x}, {y})"
    else:
        yield image, f"当前点击坐标: ({x}, {y})"
        return
    
    # 初始化客户端
    global client
    if client is None:
        loading_img = create_loading_image(original_image, "正在初始化SAM客户端...")
        yield loading_img, f"正在初始化SAM客户端...\n当前点击坐标: ({x}, {y})"
        
        client = SAMClient(server_url="http://10.30.58.120:5000")
        print('>>> client initialized')
    
    # 调用SAM模型
    loading_img = create_loading_image(original_image, "正在调用SAM模型进行分割...")
    yield loading_img, f"正在调用SAM模型进行分割...\n当前点击坐标: ({x}, {y})"
    
    selected_mask = SAM_tool(client, IMAGE_PATH, [[x, y]])
    print('>>> selected_mask: ', selected_mask.shape)
    
    # 处理图像
    loading_img = create_loading_image(original_image, "正在应用mask叠加...")
    yield loading_img, f"正在应用mask叠加...\n当前点击坐标: ({x}, {y})"
    
    img_array = original_image.copy()
    info_text = f"当前点击坐标: ({x}, {y})"
    
    img_array = apply_mask_overlay(img_array, selected_mask)
    
    # 绘制标记点
    loading_img = create_loading_image(original_image, "正在绘制标记点...")
    yield loading_img, f"正在绘制标记点...\n当前点击坐标: ({x}, {y})"
    
    # 在点击位置画红色圆点标记
    h, w = img_array.shape[:2]
    if 0 <= x < w and 0 <= y < h:
        radius = 8
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h and dx*dx + dy*dy <= radius*radius:
                    img_array[ny, nx] = [255, 0, 0]  # 红色标记
    
    # 返回最终结果
    yield img_array, f"处理完成\n当前点击坐标: ({x}, {y})"


def clear_points():
    """清除标记点"""
    global latest_click_point, original_image
    latest_click_point = None
    # 返回原始图片，清除所有标记
    if original_image is not None:
        return original_image.copy(), "标记点已清除"
    else:
        return None, "标记点已清除"

def load_image_from_path():
    """从固定路径加载图片"""
    global latest_click_point, original_image
    
    # 检查文件是否存在
    if not os.path.exists(IMAGE_PATH):
        return None, f"图片文件不存在: {IMAGE_PATH}"
    
    try:
        # 清除之前的点击记录
        latest_click_point = None
        
        # 加载并保存原始图片
        img = Image.open(IMAGE_PATH)
        
        # 处理不同格式的图片
        if img.mode == 'RGBA':
            # 将RGBA转换为RGB
            background = Image.new('RGB', img.size, (255, 255, 255))  # 白色背景
            background.paste(img, mask=img.split()[-1])  # 使用alpha通道作为mask
            img = background
        elif img.mode not in ['RGB', 'L']:
            # 其他格式转换为RGB
            img = img.convert('RGB')
        
        original_image = np.array(img)
        
        # 加载所有mask文件
        mask_count = load_masks()
        
        current_time = time.strftime("%H:%M:%S")
        info_text = f"图片已加载，尺寸: {img.size[0]}x{img.size[1]} 像素\n路径: {IMAGE_PATH}\n格式: {img.mode}\n加载了 {mask_count} 个mask文件\n更新时间: {current_time}"
        
        return original_image.copy(), info_text
        
    except Exception as e:
        return None, f"加载图片失败: {str(e)}"

def trigger_external_reload():
    """外部进程调用此函数来触发重新加载
    
    使用方法：
    在外部进程中创建文件来触发重新加载：
    
    # Python代码示例
    with open("reload_trigger.txt", "w") as f:
        f.write("reload")
    
    # 或者使用命令行
    # echo "reload" > reload_trigger.txt
    """
    try:
        with open(RELOAD_TRIGGER_FILE, "w") as f:
            f.write(f"reload_request_{time.time()}")
        print(f"重新加载触发文件已创建: {RELOAD_TRIGGER_FILE}")
        return True
    except Exception as e:
        print(f"创建触发文件失败: {str(e)}")
        return False

def check_reload_trigger():
    """检查是否需要重新加载"""
    if os.path.exists(RELOAD_TRIGGER_FILE):
        try:
            # 删除触发文件
            os.remove(RELOAD_TRIGGER_FILE)
            print("检测到重新加载请求，正在重新加载...")
            return True
        except Exception as e:
            print(f"删除触发文件失败: {str(e)}")
    return False

def auto_reload_checker():
    """自动检查重新加载的后台线程"""
    global image_display_ref, coordinate_info_ref
    
    while True:
        try:
            if check_reload_trigger():
                # 重新加载图片和masks
                new_image, new_info = load_image_from_path()
                
                # 更新Gradio组件（需要在主线程中执行）
                if image_display_ref is not None and coordinate_info_ref is not None:
                    # 使用Gradio的更新机制
                    print("重新加载完成")
                
            time.sleep(1)  # 每秒检查一次
        except Exception as e:
            print(f"自动重新加载检查出错: {str(e)}")
            time.sleep(5)

# 预加载图片以获取初始状态
def get_initial_state():
    """获取初始状态"""
    return load_image_from_path()

# 创建Gradio界面
with gr.Blocks(title="图片点击坐标标记器") as demo:
    gr.Markdown("# 图片点击坐标标记器")
    gr.Markdown("在图片上点击来标记位置并记录像素坐标。点击位置如果有对应的mask，会以绿色半透明显示。")
    gr.Markdown("**外部进程重新加载**: 创建文件 `reload_trigger.txt` 可触发重新加载")
    
    # 获取初始图片和信息
    initial_image, initial_info = get_initial_state()
    
    with gr.Row():
        with gr.Column():            
            # 显示点击坐标信息
            coordinate_info = gr.Textbox(
                label="坐标信息", 
                value=initial_info,
                lines=6,
                interactive=False
            )
            
            # 清除按钮
            clear_btn = gr.Button("清除标记点")
            
            # 重新加载按钮
            reload_btn = gr.Button("重新加载图片和mask")
            
            # 外部触发重新加载按钮（用于测试）
            external_reload_btn = gr.Button("模拟外部触发重新加载")
        
        with gr.Column():
            # 图片显示组件
            image_display = gr.Image(
                label="点击图片来标记位置",
                value=initial_image,
                interactive=True,
                show_download_button=False
            )
    
    # 保存组件引用
    image_display_ref = image_display
    coordinate_info_ref = coordinate_info
    
    # 定时检查重新加载触发器
    def check_and_reload():
        if check_reload_trigger():
            return load_image_from_path()
        return gr.update(), gr.update()
    
    # 每3秒检查一次是否需要重新加载
    timer = gr.Timer(value=3)
    timer.tick(
        fn=check_and_reload,
        outputs=[image_display, coordinate_info]
    )
    
    # 事件绑定
    image_display.select(
        fn=handle_image_click,
        inputs=[image_display],
        outputs=[image_display, coordinate_info]
        # 移除show_progress=True，因为我们现在使用generator
    )
    
    clear_btn.click(
        fn=clear_points,
        outputs=[image_display, coordinate_info]
    )
    
    reload_btn.click(
        fn=load_image_from_path,
        outputs=[image_display, coordinate_info]
    )
    
    external_reload_btn.click(
        fn=lambda: (trigger_external_reload(), "已触发外部重新加载")[1],
        outputs=[coordinate_info]
    )


# 启动应用
if __name__ == "__main__":
    demo.launch(share=False, server_name="127.0.0.1", server_port=7863) 