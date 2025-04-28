import os
import gc
import time
import folder_paths
from comfy.comfy_types import FileLocator

import torch
import torch.distributed as dist
import json

from lightx2v.utils.envs import *
from lightx2v.utils.utils import seed_all
from lightx2v.utils.profiler import ProfilingContext
from lightx2v.utils.set_config import set_config
from lightx2v.utils.registry_factory import RUNNER_REGISTER

from lightx2v.models.runners.hunyuan.hunyuan_runner import HunyuanRunner
from lightx2v.models.runners.wan.wan_runner import WanRunner
from lightx2v.models.runners.graph_runner import GraphRunner


class FakeArgs:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class Lightx2vPipeline:
    CATEGORY = "Lightx2v"
    RETURN_TYPES = ()
    FUNCTION = "pipeline"
    OUTPUT_NODE = True

    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""
        self.runner = None
        self.config = None
        self.config_dir = "custom_nodes/lightx2v_comfyui_node/lightx2v/configs"

    @classmethod    
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        img_files, _ = folder_paths.recursive_search(input_dir)
        img_files = folder_paths.filter_files_extensions(img_files, ['.jpg', '.jpeg', '.png'])

        config_dir = "custom_nodes/lightx2v_comfyui_node/lightx2v/configs"
        config_files, _ = folder_paths.recursive_search(config_dir)
        config_files = folder_paths.filter_files_extensions(config_files, ['.json'])
        print("init img_files:", img_files)
        print("init config_files:", config_files)

        data = {
            "required": {
                "model_cls": (["wan2.1", "hunyuan"],),
                "task": (["t2v", "i2v"],),
                "model_path": ([
                    "/x2v_models/wan/Wan2.1-T2V-1.3B",
                    "/x2v_models/hunyuan/lightx2v_format/t2v",
                    "/x2v_models/wan/Wan2.1-I2V-14B-480P",
                    "/x2v_models/hunyuan/lightx2v_format/i2v",
                ],),
                "prompt": ("STRING",),
                "negative_prompt": ("STRING",),
                "image": (["none",] + img_files, {"image_upload": True}),
                "config_json": (config_files,),
            }
        }
        return data

    def pipeline(
        self,
        model_cls,
        task,
        model_path,
        prompt,
        negative_prompt,
        image,
        config_json,
    ):
        image_path = None
        if image != "none":
            image_path = folder_paths.get_annotated_filepath(image) 

        full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(f"lightx2v_{task}_{model_cls}", self.output_dir)
        out_filename = f"{filename}_{counter:05}_.mp4"
        save_video_path = os.path.join(full_output_folder, out_filename)
        print("save_video_path:", save_video_path)

        args = FakeArgs(
            model_cls=model_cls,
            task=task,
            model_path=model_path,
            prompt=prompt,
            negative_prompt=negative_prompt,
            image_path=image_path,
            config_json=os.path.join(self.config_dir, config_json),
            save_video_path=save_video_path,
        )
        self.gen_video(args)

        results: list[FileLocator] = [{
            "filename": out_filename,
            "subfolder": subfolder,
            "type": self.type
        }]
        return {"ui": {"images": results, "animated": (True,)}}

    def check_same_model(self, config):
        if self.config is None \
           or self.config.model_path != config.model_path \
           or self.config.model_cls != config.model_cls \
           or self.config.task != config.task:
            return False
        return True

    def init_runner(self, config):
        print(f"config:\n{json.dumps(config, ensure_ascii=False, indent=4)}")
        if not self.check_same_model(config):
            self.config = config
            print("new model, reload...")
            if self.runner:
                self.runner = None
                gc.collect()
                torch.cuda.empty_cache()
            if CHECK_ENABLE_GRAPH_MODE():
                default_runner = RUNNER_REGISTER[config.model_cls](config)
                self.runner = GraphRunner(default_runner)
            else:
                self.runner = RUNNER_REGISTER[config.model_cls](config)
        else:
            print("same model, reload config...")
            self.runner.config = config

    def gen_video(self, args):
        with ProfilingContext("Total Cost"):
            config = set_config(args)

            seed_all(config.seed)

            if config.parallel_attn_type:
                dist.init_process_group(backend="nccl")

            self.init_runner(config)
            self.runner.run_pipeline()


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "Lightx2vPipeline": Lightx2vPipeline,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "Lightx2vPipeline": "Lightx2vPipeline",
}

