# from segment_anything import SamPredictor, sam_model_registry


# sam = sam_model_registry["vit_h"](checkpoint="<path/to/checkpoint>")
# predictor = SamPredictor(sam)
# predictor.set_image(<your_image>)
# masks, _, _ = predictor.predict(<input_prompts>)


import os
import cv2
import numpy as np
from flask import Flask, request, jsonify
import base64
import io
from PIL import Image
import torch
from segment_anything import sam_model_registry, SamPredictor

app = Flask(__name__)

# 初始化SAM模型
def initialize_sam():
    # 下载SAM模型权重文件到本地
    # 这里使用vit_h模型，您可以根据需要选择其他版本
    model_type = "vit_h"
    sam_checkpoint = "./segment-anything/sam_vit_h_4b8939.pth"  # 需要下载对应的权重文件
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    return predictor

# 全局变量存储SAM预测器
predictor = None

@app.route('/initialize', methods=['POST'])
def initialize():
    """初始化SAM模型"""
    global predictor
    try:
        if predictor is not None:
            return jsonify({"status": "success", "message": "SAM模型已初始化"})
        else:
            predictor = initialize_sam()
            return jsonify({"status": "success", "message": "SAM模型初始化成功"})
    except Exception as e:
        return jsonify({"status": "error", "message": f"初始化失败: {str(e)}"})

@app.route('/segment', methods=['POST'])
def segment_image():
    """处理图像分割请求"""
    global predictor
    
    if predictor is None:
        return jsonify({"status": "error", "message": "SAM模型未初始化"})
    
    try:
        # 获取请求数据
        data = request.json
        image_data = data.get('image')
        prompt_points = data.get('prompt_points', [])
        prompt_labels = data.get('prompt_labels', [])
        
        # 解码base64图像
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # 设置图像到预测器
        predictor.set_image(image)
        
        # 执行分割
        if prompt_points:
            input_points = np.array(prompt_points)
            input_labels = np.array(prompt_labels) if prompt_labels else np.ones(len(prompt_points))
            
            masks, scores, logits = predictor.predict(
                point_coords=input_points,
                point_labels=input_labels,
                multimask_output=True,
            )
        else:
            # 如果没有提供点击点，返回错误
            return jsonify({"status": "error", "message": "需要提供prompt点"})
        
        # 选择最佳mask（得分最高的）
        best_mask = masks[np.argmax(scores)]
        
        # 将mask转换为base64
        mask_uint8 = (best_mask * 255).astype(np.uint8)
        _, buffer = cv2.imencode('.png', mask_uint8)
        mask_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            "status": "success",
            "mask": mask_base64,
            "score": float(scores[np.argmax(scores)])
        })
        
    except Exception as e:
        return jsonify({"status": "error", "message": f"分割失败: {str(e)}"})

@app.route('/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    return jsonify({"status": "healthy", "model_loaded": predictor is not None})

if __name__ == '__main__':
    print("启动SAM服务器...")
    print("请确保已下载SAM模型权重文件")
    print("服务器运行在 http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=True) 