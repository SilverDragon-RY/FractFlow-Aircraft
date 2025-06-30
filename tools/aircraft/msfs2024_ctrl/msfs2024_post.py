import requests
import json
import math

# API端点
url = "http://10.4.147.50:5000/set"

# 要设置的参数
param_name = "PLANE_HEADING_DEGREES_TRUE"
param_value = math.radians(360)

# 准备请求数据
payload = {
    "name": param_name,
    "val": param_value
}

try:
    # 发送PUT请求到API
    response = requests.put(url, json=payload)
    
    # 检查响应状态码
    if response.status_code == 200:
        # 解析JSON响应
        data = response.json()
        print(f"设置成功: {data['message']}")
    else:
        print(f"请求失败，状态码: {response.status_code}")
        if response.text:
            try:
                error_data = response.json()
                print(f"错误信息: {error_data['message']}")
            except:
                print(f"响应内容: {response.text}")
        
except requests.exceptions.RequestException as e:
    print(f"请求错误: {e}")
except json.JSONDecodeError:
    print("无法解析JSON响应")
except Exception as e:
    print(f"发生错误: {e}")