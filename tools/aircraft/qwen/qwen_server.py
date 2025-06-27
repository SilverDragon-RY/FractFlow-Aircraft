from modelscope import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor, snapshot_download
from qwen_vl_utils import process_vision_info
import torch
import os
import cv2
import numpy as np
from flask import Flask, request, jsonify
import base64
import io
from PIL import Image


app = Flask(__name__)

# 全局变量存储Qwen预测器
model = None
processor = None
device = None

def initialize_qwen():
    """初始化Qwen模型"""
    global model, processor, device
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    try:
        # 加载模型
        model_dir = snapshot_download(
            'Qwen/Qwen2.5-VL-7B-Instruct',
            cache_dir='/data2/modelscope'
        )
        # Qwen/Qwen2.5-VL-7B-Instruct
        # Qwen/Qwen2.5-VL-32B-Instruct


        # model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        #     model_dir, 
        #     torch_dtype="auto", 
        #     device_map="auto",
        #     trust_remote_code=True
        # )
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_dir,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
        
        # 加载处理器
        processor = AutoProcessor.from_pretrained(
            model_dir,
            use_fast=True,  # 明确设置使用快速处理器
            trust_remote_code=True
        )
        
        print("Qwen模型初始化成功")
        return True
        
    except Exception as e:
        print(f"模型初始化失败: {str(e)}")
        model = None
        processor = None
        return False

@app.route('/initialize', methods=['POST'])
def initialize():
    """初始化Qwen模型"""
    global model, processor
    try:
        if model is not None and processor is not None:
            return jsonify({"status": "success", "message": "Qwen模型已初始化"})
        else:
            success = initialize_qwen()
            if success:
                return jsonify({"status": "success", "message": "Qwen模型初始化成功"})
            else:
                return jsonify({"status": "error", "message": "Qwen模型初始化失败"})
    except Exception as e:
        return jsonify({"status": "error", "message": f"初始化失败: {str(e)}"})

@app.route('/generate', methods=['POST'])
def generate_text():
    """处理文本生成请求"""
    global model, processor
    
    if model is None or processor is None:
        return jsonify({"status": "error", "message": "Qwen模型未初始化"})
    
    try:
        # 获取请求数据
        data = request.json
        image_data = data.get('image')  # base64编码的图片
        text_prompt = data.get('text', "Describe this image.")
        system_prompt = data.get('system_prompt', "You are a helpful assistant.")
        max_new_tokens = data.get('max_new_tokens', 128)
        
        # 构建消息格式
        messages = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": system_prompt},
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": f"data:image/jpeg;base64,{image_data}",
                    },
                    {"type": "text", "text": text_prompt},
                ],
            }
        ]
        
        # 准备推理
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(device)  # 使用全局device变量
        
        # 执行推理
        with torch.no_grad():  # 添加无梯度上下文
            generated_ids = model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens,
                do_sample=False,  # 确定性生成
                pad_token_id=processor.tokenizer.eos_token_id
            )
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        return jsonify({
            "status": "success",
            "output_text": output_text[0] if output_text else "",
            "prompt": text_prompt,
            "system_prompt": system_prompt
        })
        
    except Exception as e:
        return jsonify({"status": "error", "message": f"生成失败: {str(e)}"})

@app.route('/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    return jsonify({
        "status": "healthy", 
        "model_loaded": model is not None and processor is not None
    })

if __name__ == '__main__':
    print("启动Qwen服务器...")
    print("请确保已安装所需依赖")
    
    # 在启动时初始化模型
    print("正在初始化Qwen模型...")
    if initialize_qwen():
        print("模型初始化成功，服务器准备就绪")
    else:
        print("警告：模型初始化失败，请检查配置")
    
    print("服务器运行在 http://localhost:5001")
    app.run(host='0.0.0.0', port=5001, debug=False)  # 关闭debug模式避免重载问题 