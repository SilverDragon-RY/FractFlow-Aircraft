# ComfyUI Workflow Management

This document provides guidelines for creating and managing ComfyUI workflows.

## Workflow Creation Guide

### 1. Workflow File Structure

Each workflow file (.json) should contain two main sections:

```json
{
    "meta": {
        "name": "workflow_name",
        "description": "Detailed description of the workflow",
        "use_when": ["use case 1", "use case 2"],
        "input_nodes": {
            "parameter_name": {
                "node_id": "node_id",
                "field": "field_path",
                "type": "parameter_type",
                "description": "parameter description",
                "required": true/false,
                "default": "default_value (optional)"
            }
        },
        "output_nodes": {
            "output_name": {
                "node_id": "node_id",
                "type": "output_type",
                "description": "output description"
            }
        }
    },
    "workflow": {
        // Workflow JSON exported from ComfyUI in API mode
    }
}
```

### 2. Meta Section Description

- `name`: Unique identifier for the workflow
- `description`: Detailed description of the workflow
- `use_when`: List of applicable scenarios
- `input_nodes`: Input parameter definitions
  - `node_id`: ComfyUI node ID
  - `field`: Parameter path in the node (e.g., "inputs.text")
  - `type`: Parameter type (string/integer/float/boolean/number)
  - `description`: Parameter description
  - `required`: Whether the parameter is required
  - `default`: Default value (optional)
- `output_nodes`: Output node definitions
  - `node_id`: ComfyUI node ID
  - `type`: Output type (e.g., "images")
  - `description`: Output description

### 3. Workflow Section

Export the workflow JSON from ComfyUI interface and paste it directly into the `workflow` field.

### 4. Example

```json
{
    "meta": {
        "name": "text_to_image",
        "description": "Generate high-quality images based on text prompts",
        "use_when": [
            "When user only provides text description for image generation",
            "When creating new image content",
            "When generating original content without reference images"
        ],
        "input_nodes": {
            "positive_prompt": {
                "node_id": "6",
                "field": "inputs.text",
                "type": "string",
                "description": "Positive prompt describing the desired content (English required)",
                "required": true
            },
            "negative_prompt": {
                "node_id": "7",
                "field": "inputs.text",
                "type": "string",
                "description": "Negative prompt describing unwanted content (English required)",
                "default": "bad hands, bad quality, blurry"
            },
            "seed": {
                "node_id": "3",
                "field": "inputs.seed",
                "type": "integer",
                "description": "Random seed, 0 for random generation"
            }
        },
        "output_nodes": {
            "generated_image": {
                "node_id": "9",
                "type": "images",
                "description": "Main generated image file"
            }
        }
    },
    "workflow": {
        // Workflow JSON exported from ComfyUI in API mode
    }
}
```

## ComfyUI Server Setup

1. Start the ComfyUI server
2. Get the server address (typically `http://127.0.0.1:8188`)
3. Add to the `.env` file in the project root:

```env
COMFYUI_SERVER_ADDRESS=http://127.0.0.1:8188
```

## Workflow Management

Workflow files should be placed in the `workflows` directory, with one JSON file per workflow. The system will automatically scan and load all available workflows.

## Usage Recommendations

1. Provide clear descriptions and use cases for each workflow
2. Clearly define the type and necessity of all input parameters
3. Keep workflow file names concise and descriptive
4. Regularly update workflows to adapt to new requirements 