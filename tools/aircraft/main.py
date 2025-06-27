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
        self.sam_tool = SAM_TOOL(mask_type="boundary", crop_size=256)
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
                    self.image_display = gr.Image(
                        label="点击图片来标记位置",
                        value=self.current_image,
                        interactive=True,
                        show_download_button=False
                    )
            with gr.Row():
                with gr.Column():   
                    # 显示点击坐标信息
                    self.coordinate_info = gr.Textbox(
                        label="视觉分析窗口", 
                        value="无",
                        lines=6,
                        interactive=False
                    )
                    # 重新加载按钮
                    self.reload_btn = gr.Button("重新加载图片（下一帧）")     
                    # 外部触发重新加载按钮（用于测试）
                    self.external_reload_btn = gr.Button("重新加载图片")
                    # VLM 按钮
                    self.safety_vlm_btn = gr.Button("SafetyVLM分析")
                    self.brain_vlm_btn = gr.Button("BrainVLM控制")
                
                with gr.Column():
                    # 图片显示组件 1, 2
                    
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
            
            self.safety_vlm_btn.click(
                fn=self.safety_agent.run_local,
                inputs=None,
                outputs=[self.coordinate_info]
            )

        self.interface = interface

    def launch(self):
        self.interface.queue().launch(share=self.share, server_name=self.server_name, server_port=self.server_port)
# 启动应用
if __name__ == "__main__":
    my_gradio = Gradio_Interface()
    my_gradio.launch()