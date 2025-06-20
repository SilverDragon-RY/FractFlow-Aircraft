import websocket
import uuid
import json
import urllib.request
import urllib.parse
import os
from typing import List
from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv
from pathlib import Path

try:
    from .workflow_manager import WorkflowManager
except ImportError:
    from workflow_manager import WorkflowManager

load_dotenv()
mcp = FastMCP("comfyui")
server_address = os.getenv('COMFYUI_SERVER_ADDRESS', "127.0.0.1:8188")


def queue_prompt_to_comfyui(workflow: dict, client_id: str = None) -> str:
    """提交工作流到ComfyUI队列"""
    if client_id is None:
        client_id = str(uuid.uuid4())
    
    p = {"prompt": workflow, "client_id": client_id}
    data = json.dumps(p).encode('utf-8')
    req = urllib.request.Request(f"http://{server_address}/prompt", data=data)
    response = urllib.request.urlopen(req)
    result = json.loads(response.read())
    return result['prompt_id']


def wait_for_completion(prompt_id: str, client_id: str = None) -> dict:
    """等待工作流执行完成并获取结果"""
    if client_id is None:
        client_id = str(uuid.uuid4())
    
    ws = websocket.WebSocket()
    ws.connect(f"ws://{server_address}/ws?clientId={client_id}")
    
    while True:
        out = ws.recv()
        if isinstance(out, str):
            message = json.loads(out)
            if message['type'] == 'executing':
                data = message['data']
                if data['node'] is None and data['prompt_id'] == prompt_id:
                    break
    
    ws.close()
    with urllib.request.urlopen(f"http://{server_address}/history/{prompt_id}") as response:
        return json.loads(response.read())


def get_file_from_comfyui(filename: str, subfolder: str, folder_type: str) -> bytes:
    """从ComfyUI服务器获取文件内容"""
    data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
    url_values = urllib.parse.urlencode(data)
    with urllib.request.urlopen(f"http://{server_address}/view?{url_values}") as response:
        return response.read()


def download_outputs(history: dict, save_directory: str, meta: dict, custom_filename: str = None) -> List[str]:
    """下载工作流输出文件到指定目录，支持自定义文件名"""
    save_dir = Path(save_directory)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    saved_files = []
    output_nodes = meta.get('output_nodes', {})
    
    if not output_nodes:
        raise ValueError("Meta must contain 'output_nodes' definition")
    
    for output_name, output_info in output_nodes.items():
        node_id = output_info['node_id']
        output_type = output_info.get('type', 'images')
        
        if node_id not in history['outputs']:
            print(f"Warning: Output node '{node_id}' ({output_name}) not found in history")
            continue
        
        node_output = history['outputs'][node_id]
        
        if output_type in node_output:
            file_list = node_output[output_type]
            
            for i, file_info in enumerate(file_list):
                try:
                    file_data = get_file_from_comfyui(
                        file_info['filename'], file_info['subfolder'], file_info['type']
                    )
                    
                    # 智能文件命名逻辑
                    actual_extension = Path(file_info['filename']).suffix
                    
                    if custom_filename:
                        # 使用用户指定的文件名
                        custom_path = Path(custom_filename)
                        custom_stem = custom_path.stem
                        
                        # 保持实际文件的扩展名，以确保兼容性
                        if len(file_list) > 1:
                            filename = f"{custom_stem}_{i}{actual_extension}"
                        else:
                            filename = f"{custom_stem}{actual_extension}"
                    else:
                        # 使用默认的语义化命名
                        filename = f"{output_name}_{i}{actual_extension}" if len(file_list) > 1 else f"{output_name}{actual_extension}"
                    
                    file_path = save_dir / filename
                    with open(file_path, 'wb') as f:
                        f.write(file_data)
                    
                    saved_files.append(str(file_path))
                    print(f"Downloaded: {filename} ({output_type})")
                    
                except Exception as e:
                    print(f"Warning: Failed to download {output_type} from node {node_id}: {e}")
        else:
            print(f"Warning: Output type '{output_type}' not found for node '{node_id}' ({output_name})")
    
    return saved_files


@mcp.tool()
async def list_comfyui_workflows() -> str:
    """列出所有可用的ComfyUI工作流及其完整文档"""
    try:
        workflow_manager = WorkflowManager()
        return workflow_manager.generate_workflow_docs()
    except Exception as e:
        return f"Error listing workflows: {str(e)}"


@mcp.tool()
async def execute_comfyui_workflow(workflow_name: str, save_path: str, parameters: dict = None) -> str:
    """
    执行指定的ComfyUI工作流并保存结果
    
    Args:
        workflow_name: 要执行的工作流名称
        save_path: 输出文件保存路径（目录或完整文件路径）
        parameters: 工作流所需的参数字典
    """
    try:
        if parameters is None:
            parameters = {}
        
        # 规范化保存路径
        expanded_path = os.path.expanduser(save_path)
        if not os.path.isabs(expanded_path):
            expanded_path = os.path.abspath(expanded_path)
        save_path = expanded_path
        
        # 智能路径解析：检测用户是否指定了文件名
        if not os.path.isdir(save_path) and ('.' in os.path.basename(save_path)):
            # 用户指定了文件路径
            save_directory = os.path.dirname(save_path) or "."
            custom_filename = os.path.basename(save_path)
        else:
            # 用户指定了目录路径
            save_directory = save_path
            custom_filename = None
        
        workflow_manager = WorkflowManager()
        
        # 加载工作流
        try:
            meta, workflow = workflow_manager.load_workflow(workflow_name)
        except FileNotFoundError:
            available = workflow_manager.get_available_workflows()
            return f"Workflow '{workflow_name}' not found. Available workflows: {', '.join(available)}"
        
        # 验证参数
        validation_errors = workflow_manager.validate_parameters(meta, **parameters)
        if validation_errors:
            return f"Parameter validation failed:\n" + "\n".join(f"- {error}" for error in validation_errors)
        
        # 填充默认值
        input_nodes = meta.get("input_nodes", {})
        for param_name, param_info in input_nodes.items():
            if param_name not in parameters and 'default' in param_info:
                parameters[param_name] = param_info['default']
        
        # 填充参数到工作流并执行
        filled_workflow = workflow_manager.fill_parameters(workflow, meta, **parameters)
        client_id = str(uuid.uuid4())
        prompt_id = queue_prompt_to_comfyui(filled_workflow, client_id)
        history = wait_for_completion(prompt_id, client_id)
        
        # 下载文件，传递自定义文件名
        saved_files = download_outputs(history[prompt_id], save_directory, meta, custom_filename)
        
        if not saved_files:
            return f"Workflow '{workflow_name}' executed successfully but no output files were generated."
        
        # 构建结果报告
        result = f"Workflow '{workflow_name}' executed successfully!\n"
        result += f"Generated {len(saved_files)} output file(s):\n"
        for file_path in saved_files:
            result += f"- {file_path}\n"
        
        if parameters:
            result += f"\nUsed parameters:\n"
            for param_name, param_value in parameters.items():
                result += f"- {param_name}: {param_value}\n"
        
        return result.strip()
        
    except Exception as e:
        return f"Error executing workflow '{workflow_name}': {str(e)}"


if __name__ == "__main__":
    mcp.run(transport='stdio') 