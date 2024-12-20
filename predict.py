import os
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
            weights_to_download=[
                "ltx-video-2b-v0.9.1.safetensors",
            ],
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
        model_loader = workflow["44"]["inputs"]
        model_loader["ckpt_name"] = f"ltx-video-2b-v{kwargs['model']}.safetensors"

        if not kwargs["image_filename"]:
            del workflow["77"]
            del workflow["78"]
            del workflow["81"]
            del workflow["82"]
            workflow["72"]["inputs"]["latent_image"] = ["84", 0]
            workflow["71"]["inputs"]["latent"] = ["84", 0]
            workflow["69"]["inputs"]["positive"] = ["6", 0]
            workflow["69"]["inputs"]["negative"] = ["7", 0]

            aspect_ratio_size = workflow["85"]["inputs"]
            aspect_ratio_size["target_size"] = kwargs["target_size"]
            aspect_ratio_size["aspect_ratio"] = kwargs["aspect_ratio"]

            length = workflow["84"]["inputs"]
            length["length"] = kwargs["length"]
        else:
            del workflow["84"]
            del workflow["85"]

            target_size = workflow["81"]["inputs"]
            target_size["target_size"] = kwargs["target_size"]

            img_to_video = workflow["77"]["inputs"]
            img_to_video["length"] = kwargs["length"]

            # Update input image
            if kwargs["image_filename"]:
                load_image = workflow["78"]["inputs"]
                load_image["image"] = kwargs["image_filename"]

        # Update positive prompt
        positive_prompt = workflow["6"]["inputs"]
        positive_prompt["text"] = kwargs["prompt"]

        # Update negative prompt
        negative_prompt = workflow["7"]["inputs"]
        negative_prompt["text"] = kwargs["negative_prompt"]

        # Update cfg scale
        sampler = workflow["72"]["inputs"]
        sampler["cfg"] = kwargs["cfg_scale"]
        sampler["noise_seed"] = kwargs["seed"]

        # Update steps
        scheduler = workflow["71"]["inputs"]
        scheduler["steps"] = kwargs["steps"]

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
            description="Optional input image to use as the starting frame",
            default=None,
        ),
        target_size: int = Input(
            description="Target size for the output video",
            default=640,
            choices=[512, 576, 640, 704, 768, 832, 896, 960, 1024],
        ),
        aspect_ratio: str = Input(
            description="Aspect ratio of the output video. Ignored if an image is provided.",
            default="3:2",
            choices=[
                "1:1",
                "1:2",
                "2:1",
                "2:3",
                "3:2",
                "3:4",
                "4:3",
                "4:5",
                "5:4",
                "9:16",
                "16:9",
                "9:21",
                "21:9",
            ],
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
        model: str = Input(
            description="Model version to use",
            default="0.9.1",
            choices=["0.9.1", "0.9"],
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
            aspect_ratio=aspect_ratio,
            cfg_scale=cfg,
            steps=steps,
            length=length,
            model=model,
            seed=seed,
        )

        wf = self.comfyUI.load_workflow(workflow)
        self.comfyUI.connect()
        self.comfyUI.run_workflow(wf)

        return self.comfyUI.get_files(OUTPUT_DIR, file_extensions=["mp4"])
