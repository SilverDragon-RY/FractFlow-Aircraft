import os
import json
from typing import Dict, Tuple, List, Any, Optional
from pathlib import Path


class WorkflowManager:
    """ComfyUI工作流管理器，负责加载、验证和处理工作流配置"""
    
    def __init__(self, workflows_dir: Optional[str] = None):
        """
        初始化工作流管理器
        
        Args:
            workflows_dir: 工作流配置目录路径，默认为当前目录下的workflows
        """
        if workflows_dir is None:
            # 默认为当前文件所在目录下的workflows目录
            current_dir = Path(__file__).parent
            self.workflows_dir = current_dir / "workflows"
        else:
            self.workflows_dir = Path(workflows_dir)
        
        # 确保工作流目录存在
        self.workflows_dir.mkdir(exist_ok=True)
        
        # 缓存已加载的工作流
        self._workflow_cache = {}
    
    def scan_workflows(self) -> Dict[str, dict]:
        """
        扫描工作流目录，返回所有可用的工作流信息
        
        Returns:
            Dict[str, dict]: 工作流名称到元数据的映射
        """
        workflows = {}
        
        if not self.workflows_dir.exists():
            return workflows
        
        # 扫描所有.json文件
        for json_file in self.workflows_dir.glob("*.json"):
            try:
                workflow_name = json_file.stem
                meta, _ = self.load_workflow(workflow_name)
                workflows[workflow_name] = meta
            except Exception as e:
                print(f"Warning: Failed to load workflow {json_file}: {e}")
                continue
        
        return workflows
    
    def get_available_workflows(self) -> List[str]:
        """
        获取所有可用工作流的名称列表
        
        Returns:
            List[str]: 工作流名称列表
        """
        return list(self.scan_workflows().keys())
    
    def load_workflow(self, name: str) -> Tuple[dict, dict]:
        """
        加载指定的工作流配置
        
        Args:
            name: 工作流名称
            
        Returns:
            Tuple[dict, dict]: (meta, workflow) 元数据和工作流定义
            
        Raises:
            FileNotFoundError: 工作流文件不存在
            ValueError: 工作流格式错误
        """
        # 检查缓存
        if name in self._workflow_cache:
            return self._workflow_cache[name]
        
        workflow_file = self.workflows_dir / f"{name}.json"
        
        if not workflow_file.exists():
            raise FileNotFoundError(f"Workflow '{name}' not found at {workflow_file}")
        
        try:
            with open(workflow_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 验证基本结构
            if "meta" not in data or "workflow" not in data:
                raise ValueError(f"Workflow '{name}' missing required 'meta' or 'workflow' sections")
            
            meta = data["meta"]
            workflow = data["workflow"]
            
            # 验证meta结构
            required_meta_fields = ["name", "description"]
            for field in required_meta_fields:
                if field not in meta:
                    raise ValueError(f"Workflow '{name}' meta missing required field: {field}")
            
            # 缓存结果
            self._workflow_cache[name] = (meta, workflow)
            
            return meta, workflow
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in workflow '{name}': {e}")
        except Exception as e:
            raise ValueError(f"Failed to load workflow '{name}': {e}")
    
    def fill_parameters(self, workflow: dict, meta: dict, **params) -> dict:
        """
        根据参数填充工作流模板
        
        Args:
            workflow: 工作流定义
            meta: 工作流元数据
            **params: 要填充的参数
            
        Returns:
            dict: 填充参数后的工作流
        """
        # 深拷贝工作流，避免修改原始数据
        filled_workflow = json.loads(json.dumps(workflow))
        
        # 获取输入节点定义
        input_nodes = meta.get("input_nodes", {})
        
        # 遍历所有参数
        for param_name, param_value in params.items():
            if param_name in input_nodes:
                node_info = input_nodes[param_name]
                node_id = node_info["node_id"]
                field_path = node_info["field"]
                
                # 按路径设置值
                self._set_nested_value(filled_workflow, node_id, field_path, param_value)
        
        return filled_workflow
    
    def _set_nested_value(self, workflow: dict, node_id: str, field_path: str, value: Any):
        """
        在嵌套字典中设置值
        
        Args:
            workflow: 工作流字典
            node_id: 节点ID
            field_path: 字段路径，如 "inputs.text"
            value: 要设置的值
        """
        if node_id not in workflow:
            raise ValueError(f"Node '{node_id}' not found in workflow")
        
        # 分解路径
        path_parts = field_path.split(".")
        
        # 定位到目标位置
        target = workflow[node_id]
        for part in path_parts[:-1]:
            if part not in target:
                raise ValueError(f"Field path '{field_path}' not found in node '{node_id}'")
            target = target[part]
        
        # 设置最终值
        final_field = path_parts[-1]
        target[final_field] = value
    
    def validate_parameters(self, meta: dict, **params) -> List[str]:
        """
        验证参数是否符合工作流要求
        
        Args:
            meta: 工作流元数据
            **params: 要验证的参数
            
        Returns:
            List[str]: 错误信息列表，空列表表示验证通过
        """
        errors = []
        input_nodes = meta.get("input_nodes", {})
        
        # 检查必需参数
        for param_name, param_info in input_nodes.items():
            if param_info.get("required", False) and param_name not in params:
                errors.append(f"Missing required parameter: {param_name}")
        
        # 检查参数类型
        for param_name, param_value in params.items():
            if param_name in input_nodes:
                param_info = input_nodes[param_name]
                expected_type = param_info.get("type", "string")
                
                if not self._validate_type(param_value, expected_type):
                    errors.append(f"Parameter '{param_name}' should be of type {expected_type}, got {type(param_value).__name__}")
        
        return errors
    
    def _validate_type(self, value: Any, expected_type: str) -> bool:
        """
        验证值的类型
        
        Args:
            value: 要验证的值
            expected_type: 期望的类型字符串
            
        Returns:
            bool: 是否类型匹配
        """
        type_mapping = {
            "string": str,
            "integer": int,
            "float": float,
            "boolean": bool,
            "number": (int, float)
        }
        
        if expected_type in type_mapping:
            expected_python_type = type_mapping[expected_type]
            return isinstance(value, expected_python_type)
        
        return True  # 未知类型，跳过验证
    
    def generate_workflow_docs(self) -> str:
        """
        生成所有工作流的文档字符串
        
        Returns:
            str: 格式化的工作流文档
        """
        workflows = self.scan_workflows()
        
        if not workflows:
            return "No workflows available."
        
        docs = []
        
        for workflow_name, meta in workflows.items():
            doc = f"""
========== {meta['name']} ==========
Description: {meta['description']}"""
            
            # 使用场景
            use_when = meta.get('use_when', [])
            if use_when:
                doc += f"\nUse when: {', '.join(use_when)}"
            
            # 输入参数
            input_nodes = meta.get('input_nodes', {})
            if input_nodes:
                doc += "\n\nParameters:"
                for param_name, param_info in input_nodes.items():
                    param_type = param_info.get('type', 'string')
                    param_desc = param_info.get('description', 'No description')
                    required = " (required)" if param_info.get('required', False) else ""
                    default = param_info.get('default')
                    default_str = f" (default: {default})" if default is not None else ""
                    
                    doc += f"\n    {param_name} ({param_type}): {param_desc}{required}{default_str}"
            
            # 输出信息
            output_nodes = meta.get('output_nodes', {})
            if output_nodes:
                doc += "\n\nOutputs:"
                for output_name, output_info in output_nodes.items():
                    output_type = output_info.get('type', 'file')
                    output_desc = output_info.get('description', 'No description')
                    doc += f"\n    {output_name} ({output_type}): {output_desc}"
            
            # 使用示例
            doc += f"\n\nUsage:\n    execute_comfyui_workflow('{workflow_name}', save_path='/path/to/output', **parameters)"
            
            docs.append(doc)
        
        return "\n".join(docs) 