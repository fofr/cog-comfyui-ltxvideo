import os
import mimetypes
import json
import shutil
from typing import List
from cog import BasePredictor, Input, Path
from comfyui import ComfyUI
from cog_model_helpers import seed as seed_helper

OUTPUT_DIR = "/tmp/outputs"
INPUT_DIR = "/tmp/inputs"
COMFYUI_TEMP_OUTPUT_DIR = "ComfyUI/temp"
ALL_DIRECTORIES = [OUTPUT_DIR, INPUT_DIR, COMFYUI_TEMP_OUTPUT_DIR]

# Save your example JSON to the same directory as predict.py
api_json_file = "workflow_api.json"

# Force HF offline
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"


class Predictor(BasePredictor):
    def setup(self):
        self.comfyUI = ComfyUI("127.0.0.1:8188")
        self.comfyUI.start_server(OUTPUT_DIR, INPUT_DIR)

        # Give a list of weights filenames to download during setup
        with open(api_json_file, "r") as file:
            workflow = json.loads(file.read())
        self.comfyUI.handle_weights(
            workflow,
            weights_to_download=[],
        )

    def filename_with_extension(self, input_file, prefix):
        extension = os.path.splitext(input_file.name)[1]
        return f"{prefix}{extension}"

    def handle_input_file(
        self,
        input_file: Path,
        filename: str = "image.png",
    ):
        shutil.copy(input_file, os.path.join(INPUT_DIR, filename))

    # Update nodes in the JSON workflow to modify your workflow based on the given inputs
    def update_workflow(self, workflow, **kwargs):
        # Update positive prompt
        positive_prompt = workflow["6"]["inputs"]
        positive_prompt["text"] = kwargs["prompt"]

        # Update negative prompt
        negative_prompt = workflow["7"]["inputs"]
        negative_prompt["text"] = kwargs["negative_prompt"]

        # Update target size
        target_size = workflow["81"]["inputs"]
        target_size["target_size"] = kwargs["target_size"]

        # Update cfg scale
        sampler = workflow["72"]["inputs"]
        sampler["cfg"] = kwargs["cfg_scale"]
        sampler["noise_seed"] = kwargs["seed"]

        # Update steps
        scheduler = workflow["71"]["inputs"]
        scheduler["steps"] = kwargs["steps"]

        # Update length
        img_to_video = workflow["77"]["inputs"]
        img_to_video["length"] = kwargs["length"]

        # Update input image
        if kwargs["image_filename"]:
            load_image = workflow["78"]["inputs"]
            load_image["image"] = kwargs["image_filename"]

    def predict(
        self,
        prompt: str = Input(
            description="Text prompt for the video. This model needs long descriptive prompts, if the prompt is too short the quality won't be good.",
            default="best quality, 4k, HDR, a tracking shot of a beautiful scene",
        ),
        negative_prompt: str = Input(
            description="Things you do not want to see in your video",
            default="low quality, worst quality, deformed, distorted",
        ),
        image: Path = Input(
            description="Input image to animate",
            default=None,
        ),
        target_size: int = Input(
            description="Target size for the output video",
            default=640,
            choices=[512, 576, 640, 704, 768, 832, 896, 960, 1024],
        ),
        cfg: float = Input(
            description="How strongly the video follows the prompt",
            default=3.0,
            ge=1.0,
            le=20.0,
        ),
        steps: int = Input(
            description="Number of steps",
            default=30,
            ge=1,
            le=50,
        ),
        length: int = Input(
            description="Length of the output video in frames",
            default=97,
            choices=[97, 129, 161, 193, 225, 257],
        ),
        seed: int = seed_helper.predict_seed(),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        self.comfyUI.cleanup(ALL_DIRECTORIES)
        seed = seed_helper.generate(seed)

        image_filename = None
        if image:
            image_filename = self.filename_with_extension(image, "image")
            self.handle_input_file(image, image_filename)

        with open(api_json_file, "r") as file:
            workflow = json.loads(file.read())

        self.update_workflow(
            workflow,
            prompt=prompt,
            negative_prompt=negative_prompt,
            image_filename=image_filename,
            target_size=target_size,
            cfg_scale=cfg,
            steps=steps,
            length=length,
            seed=seed,
        )

        wf = self.comfyUI.load_workflow(workflow)
        self.comfyUI.connect()
        self.comfyUI.run_workflow(wf)

        return self.comfyUI.get_files(OUTPUT_DIR, file_extensions=["mp4"])
