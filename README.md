# FractFlow

FractFlow is a fractal intelligence architecture that decomposes intelligence into nestable Agent-Tool units, building dynamically evolving distributed cognitive systems through recursive composition.

## Design Philosophy

FractFlow is a fractal intelligence architecture that decomposes intelligence into nestable Agent-Tool units, building dynamically evolving distributed cognitive systems through recursive composition.

Each agent not only possesses cognitive capabilities but also has the ability to call other agents, forming self-referential, self-organizing, and self-adaptive intelligent flows.

Similar to how each tentacle of an octopus has its own brain in a collaborative structure, FractFlow achieves structurally malleable and behaviorally evolving distributed intelligence through the combination and coordination of modular intelligence.

## Installation

Please install "uv" first: https://docs.astral.sh/uv/#installation

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install .
```

Note: The project is still under development. If dependencies are not satisfied, please install manually: `uv pip install XXXX`.

## Aircraft Tool Regulation

1: Please put all tools under tools/aircraft/xxx

2: When calling a tool, the directory should be tools/aircraft/xxx (assume launched from root)