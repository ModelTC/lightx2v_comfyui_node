"""Top-level package for lightx2v_comfyui_node."""

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
]

__author__ = """lightx2v_comfyui_node"""
__email__ = "liangliu@buaa.edu.cn"
__version__ = "0.0.1"

from .src.lightx2v_comfyui_node.nodes import NODE_CLASS_MAPPINGS
from .src.lightx2v_comfyui_node.nodes import NODE_DISPLAY_NAME_MAPPINGS

WEB_DIRECTORY = "./web"
