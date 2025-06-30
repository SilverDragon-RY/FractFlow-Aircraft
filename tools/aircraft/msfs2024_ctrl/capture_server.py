from flask import Flask, request, jsonify
import os
from datetime import datetime
import uuid
from PIL import Image
import io
import base64

app = Flask(__name__)

# 配置上传目录
UPLOAD_FOLDER = './screenshots'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/upload', methods=['POST'])
def upload_screenshot():
    try:
        # 检查请求中是否包含图像数据
        if 'image' not in request.json:
            return jsonify({'error': '没有找到图像数据'}), 400
        
        # 获取base64编码的图像数据
        image_data = request.json['image']
        
        # 解码base64图像数据
        try:
            # 如果数据包含data:image前缀，移除它
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            
            # 解码base64数据
            image_bytes = base64.b64decode(image_data)
            
            # 使用PIL打开图像
            image = Image.open(io.BytesIO(image_bytes))
            
            # 生成唯一的文件名
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            unique_id = str(uuid.uuid4())[:8]
            # filename = f'screenshot_{timestamp}_{unique_id}.png'
            filename = 'current_view.png'
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            
            # 原子写入：先保存到临时文件，然后原子性地移动到目标文件
            temp_filename = f'temp_{unique_id}.png'
            temp_filepath = os.path.join(UPLOAD_FOLDER, temp_filename)
            
            # 保存为PNG格式到临时文件
            image.save(temp_filepath, 'PNG')
            
            # 原子性地移动到目标文件（这是原子操作，不会被中断）
            os.rename(temp_filepath, filepath)
            
            return jsonify({
                'success': True,
                'message': '图像上传成功',
                'filename': filename,
                'path': filepath,
                'size': f'{image.width}x{image.height}'
            }), 200
            
        except Exception as e:
            return jsonify({'error': f'图像处理失败: {str(e)}'}), 400
            
    except Exception as e:
        return jsonify({'error': f'服务器错误: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'server running', 'upload_folder': UPLOAD_FOLDER}), 200

if __name__ == '__main__':
    print(f"服务器启动中...")
    print(f"图像保存目录: {os.path.abspath(UPLOAD_FOLDER)}")
    print(f"服务器地址: http://10.30.58.120:5050")
    print(f"上传接口: http://10.30.58.120:5050/upload")
    app.run(host='0.0.0.0', port=5050, debug=True) 