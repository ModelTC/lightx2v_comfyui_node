import argparse
import os
import torch
import torch.distributed as dist
import time
import gc
import json
from comfy.comfy_types import FileLocator
import folder_paths
import json
from fractions import Fraction
from comfy.comfy_types import FileLocator
from lightx2v.utils.utils import save_videos_grid, seed_all, cache_video
from lightx2v.__main__ import load_models, run_image_encoder, run_text_encoder, set_target_shape, init_scheduler 
from lightx2v.utils.set_config import set_config
from lightx2v.utils.profiler import ProfilingContext, ProfilingContext4Debug
from lightx2v.models.runners.default_runner import DefaultRunner
from lightx2v.models.runners.graph_runner import GraphRunner
from lightx2v.utils.envs import *

class FakeArgs:
    def __init__(
        self,
        model_cls: str,
        task: str,
        model_path: str,
        prompt: str,
        infer_steps: int,
        target_video_length: int,
        target_width: int,
        target_height: int,
        attention_type: str,
        sample_neg_prompt: str,
        sample_guide_scale: float,
        sample_shift: float,
        config_path: str = None,
        image_path: str = None,
        save_video_path: str = "output_ligthx2v.mp4",
        do_mm_calib: bool = False,
        cpu_offload: bool = False,
        feature_caching: str = "NoCaching",
        mm_config: str = None,
        seed: int = 42,
        parallel_attn_type: str = None,
        parallel_vae: bool = False,
        max_area: bool = False,
        vae_stride: tuple = (4, 8, 8),
        patch_size: tuple = (1, 2, 2),
        teacache_thresh: float = 0.26,
        use_ret_steps: bool = False,
        use_bfloat16: bool = True,
        lora_path: str = None,
        strength_model: float = 1.0,
    ):
        self.model_cls = model_cls
        self.task = task
        self.model_path = model_path
        self.prompt = prompt
        self.infer_steps = infer_steps
        self.target_video_length = target_video_length
        self.target_width = target_width
        self.target_height = target_height
        self.attention_type = attention_type
        self.sample_neg_prompt = sample_neg_prompt
        self.sample_guide_scale = sample_guide_scale
        self.sample_shift = sample_shift
        self.seed = seed
        self.config_path = config_path
        self.image_path = image_path
        self.save_video_path = save_video_path
        self.do_mm_calib = do_mm_calib
        self.cpu_offload = cpu_offload
        self.feature_caching = feature_caching
        self.mm_config = mm_config
        self.parallel_attn_type = parallel_attn_type
        self.parallel_vae = parallel_vae
        self.max_area = max_area
        self.vae_stride = vae_stride
        self.patch_size = patch_size
        self.teacache_thresh = teacache_thresh
        self.use_ret_steps = use_ret_steps
        self.use_bfloat16 = use_bfloat16
        self.lora_path = lora_path
        self.strength_model = strength_model


def gen_video(args):

    start_time = time.perf_counter()
    print(f"args: {args}")

    seed_all(args.seed)

    config = set_config(args)

    if config.parallel_attn_type:
        dist.init_process_group(backend="nccl")

    print(f"config: {config}")

    with ProfilingContext("Load models"):
        model, text_encoders, vae_model, image_encoder = load_models(config)

    if config["task"] in ["i2v"]:
        image_encoder_output = run_image_encoder(config, image_encoder, vae_model)
    else:
        image_encoder_output = {"clip_encoder_out": None, "vae_encode_out": None}

    with ProfilingContext("Run Text Encoder"):
        text_encoder_output = run_text_encoder(config["prompt"], text_encoders, config, image_encoder_output)

    inputs = {"text_encoder_output": text_encoder_output, "image_encoder_output": image_encoder_output}

    set_target_shape(config, image_encoder_output)
    scheduler = init_scheduler(config, image_encoder_output)

    model.set_scheduler(scheduler)

    gc.collect()
    torch.cuda.empty_cache()

    if CHECK_ENABLE_GRAPH_MODE():
        default_runner = DefaultRunner(model, inputs)
        runner = GraphRunner(default_runner)
    else:
        runner = DefaultRunner(model, inputs)

    latents, generator = runner.run()

    if config.cpu_offload:
        scheduler.clear()
        del text_encoder_output, image_encoder_output, model, text_encoders, scheduler
        torch.cuda.empty_cache()

    with ProfilingContext("Run VAE"):
        images = vae_model.decode(latents, generator=generator, config=config)

    if not config.parallel_attn_type or (config.parallel_attn_type and dist.get_rank() == 0):
        with ProfilingContext("Save video"):
            if config.model_cls == "wan2.1":
                cache_video(tensor=images, save_file=config.save_video_path, fps=16, nrow=1, normalize=True, value_range=(-1, 1))
            else:
                save_videos_grid(images, config.save_video_path, fps=24)

    end_time = time.perf_counter()
    print(f"Total cost: {end_time - start_time}")


class Lightx2vPipeline:
    CATEGORY = "Lightx2v"
    RETURN_TYPES = ()
    FUNCTION = "pipeline"
    OUTPUT_NODE = True

    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""

    @classmethod    
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
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
                "infer_steps": ("INT",),
                "target_video_length": ("INT",),
                "target_width": ("INT",),
                "target_height": ("INT",),
                "attention_type": ("STRING",),
                "sample_neg_prompt": ("STRING", {"default": ""}),
                "sample_guide_scale": ("FLOAT", {"default": 5.0}),
                "sample_shift": ("FLOAT", {"default": 5.0}),
                "do_mm_calib": ("BOOLEAN", {"default": False}),
                "cpu_offload": ("BOOLEAN", {"default": False}),
                "feature_caching": (["NoCaching", "TaylorSeer", "Tea"],),
                "mm_config": ("STRING", {"default": ""}),
                "seed": ("INT", {"default": 42}),
                "parallel_attn_type": (["none", "ulysses", "ring"],),
                "parallel_vae": ("BOOLEAN", {"default": False}),
                "max_area": ("BOOLEAN", {"default": False}),
                "teacache_thresh": ("FLOAT", {"default": 0.26}),
                "use_ret_steps": ("BOOLEAN", {"default": False}),
                "image": (["none",] + sorted(files), {"image_upload": True}),
                "use_bfloat16": ("BOOLEAN", {"default": True}),
                "lora_path": ("STRING", {"default": "none"}),
                "strength_model": ("FLOAT", {"default": 1.0}),
                "vae_stride": ("STRING", {"default": "4-8-8"}),
                "patch_size": ("STRING", {"default": "1-2-2"}),
            }
        }
        return data

    def pipeline(
        self,
        model_cls,
        task,
        model_path,
        prompt,
        infer_steps,
        target_video_length,
        target_width,
        target_height,
        attention_type,
        sample_neg_prompt,
        sample_guide_scale,
        sample_shift,
        do_mm_calib,
        cpu_offload,
        feature_caching,
        mm_config,
        seed,
        parallel_attn_type,
        parallel_vae,
        max_area,
        teacache_thresh,
        use_ret_steps,
        image,
        use_bfloat16,
        lora_path,
        strength_model,
        vae_stride,
        patch_size,
    ):
        if parallel_attn_type == "none":
            parallel_attn_type = None
        if lora_path == "none":
            lora_path = None
        vae_stride = tuple([int(x) for x in vae_stride.split('-')])
        patch_size = tuple([int(x) for x in patch_size.split('-')])
        config_path = None
        if model_cls == "wan2.1":
            config_path = os.path.join(model_path, "config.json")
        image_path = None
        if image != "none":
            image_path = folder_paths.get_annotated_filepath(image) 

        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(f"lightx2v_{task}_{model_cls}", self.output_dir, target_width, target_height)
        out_filename = f"{filename}_{counter:05}_.mp4"
        save_video_path = os.path.join(full_output_folder, out_filename)

        args = FakeArgs(
            model_cls,
            task,
            model_path,
            prompt,
            infer_steps,
            target_video_length,
            target_width,
            target_height,
            attention_type,
            sample_neg_prompt,
            sample_guide_scale,
            sample_shift,
            config_path,
            image_path,
            save_video_path,
            do_mm_calib,
            cpu_offload,
            feature_caching,
            mm_config,
            seed,
            parallel_attn_type,
            parallel_vae,
            max_area,
            vae_stride,
            patch_size,
            teacache_thresh,
            use_ret_steps,
            use_bfloat16,
            lora_path,
            strength_model,
        )

        gen_video(args)

        results: list[FileLocator] = [{
            "filename": out_filename,
            "subfolder": subfolder,
            "type": self.type
        }]
        return {"ui": {"images": results, "animated": (True,)}}  # TODO: frontend side

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "Lightx2vPipeline": Lightx2vPipeline,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "Lightx2vPipeline": "Lightx2vPipeline",
}

