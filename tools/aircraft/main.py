import gradio as gr
import numpy as np
from PIL import Image
import os
import glob
import time
import threading
from functools import partial

from sam.sam_mcp import SAM_TOOL
from safety_check.safety_agent import Safety_Agent

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



class Gradio_Interface():
    def __init__(self, share=False, server_name="127.0.0.1", server_port=7863):
        # TEST VARS
        self.img_path = "./test.png"

        # True Vars
        self.share = share
        self.server_name = server_name
        self.server_port = server_port
        self.current_image = None

        # --- init all components ---
        self.sam_tool = SAM_TOOL(mask_type="boundary", crop_size=1024)
        print("SAM Component Loaded !")
        self.safety_agent = Safety_Agent()
        print("Safety VLM Component Loaded !")
        # ---------------------------

        # init gradio IO after all other components
        self._build_interface()
    
    # Gradio Interface Core
    def _build_interface(self):
        with gr.Blocks(title="自动降落程序 - 示例") as interface:
            gr.Markdown("# 图片点击坐标标记器")
            gr.Markdown("**外部进程重新加载**: 创建文件 `reload_trigger.txt` 可触发重新加载")
            # 获取初始图片
            self.current_image = self.sam_tool.load_frame(self.img_path)
            with gr.Row():
                with gr.Column():   
                    # 显示点击坐标信息
                    self.coordinate_info = gr.Textbox(
                        label="信息", 
                        value="无",
                        lines=6,
                        interactive=False
                    )
                    # 重新加载按钮
                    self.reload_btn = gr.Button("重新加载图片（下一帧）")     
                    # 外部触发重新加载按钮（用于测试）
                    self.external_reload_btn = gr.Button("模拟外部触发重新加载")
                    # VLM 按钮
                    self.vlm_btn = gr.Button("模拟VLM分析")
                
                with gr.Column():
                    # 图片显示组件 1, 2
                    self.image_display = gr.Image(
                        label="点击图片来标记位置",
                        value=self.current_image,
                        interactive=True,
                        show_download_button=False
                    )
                    self.image_to_vlm = gr.Image(
                        label="VLM看到的图片",
                        value=None,
                        interactive=False,
                        show_download_button=False
                    )
            
            # 保存组件引用
            self.image_display_ref = self.image_display
            self.coordinate_info_ref = self.coordinate_info
            self.image_to_vlm_ref = self.image_to_vlm
            
            # 定时检查重新加载触发器
            def check_and_reload():
                if check_reload_trigger():
                    return self.sam_tool.load_frame(self.img_path)
                return gr.update(), gr.update()
            
            # 每3秒检查一次是否需要重新加载
            timer = gr.Timer(value=3)
            timer.tick(
                fn=check_and_reload,
                outputs=[self.image_display, self.coordinate_info]
            )
            
            # 事件绑定
            self.image_display.select(
                fn=self.sam_tool.detect,
                inputs=[self.image_display],
                outputs=[self.image_display, self.image_to_vlm, self.coordinate_info]
            )
            
            self.reload_btn.click(
                fn=lambda: self.sam_tool.load_frame(self.img_path),
                inputs=None,
                outputs=[self.image_display]
            )
            
            self.external_reload_btn.click(
                fn=lambda: (trigger_external_reload(), "已触发外部重新加载")[1],
                outputs=[self.coordinate_info, self.coordinate_info]
            )
        self.interface = interface

    def launch(self):
        self.interface.queue().launch(share=self.share, server_name=self.server_name, server_port=self.server_port)
# 启动应用
if __name__ == "__main__":
    my_gradio = Gradio_Interface()
    my_gradio.launch()