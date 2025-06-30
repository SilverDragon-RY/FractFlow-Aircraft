import requests
import json

# API端点
url = "http://10.4.147.50:5000/get"

try:
    # 发送GET请求到API
    response = requests.get(url)
    
    # 检查响应状态码
    if response.status_code == 200:
        # 解析JSON响应
        data = response.json()
        
        # 打印所有参数
        print("飞机状态参数:")
        for param in data:
            print(f"{param['name']}: {param['val']} {param['unit']} (可写: {param['writable']})")
        
        # 示例：提取特定参数
        for param in data:
            if param['name'] == 'PLANE_ALTITUDE':
                altitude = param['val']
                print(f"\n当前高度: {altitude} 英尺")
            elif param['name'] == 'PLANE_HEADING_DEGREES_TRUE':
                heading = param['val']
                print(f"当前真航向: {heading} 度")
            elif param['name'] == 'VELOCITY_BODY_X':
                speed = param['val']
                print(f"当前前向速度: {speed} 英尺/秒")
    else:
        print(f"请求失败，状态码: {response.status_code}")
        
except requests.exceptions.RequestException as e:
    print(f"请求错误: {e}")
except json.JSONDecodeError:
    print("无法解析JSON响应")
except Exception as e:
    print(f"发生错误: {e}")
