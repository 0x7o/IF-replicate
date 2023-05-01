from deepfloyd_if.modules import IFStageI, IFStageII, StableStageIII
from deepfloyd_if.modules.t5 import T5Embedder
from cog import BasePredictor, Path, Input
from deepfloyd_if.pipelines import dream
from typing import List
import huggingface_hub
import random
import json


class Predictor(BasePredictor):
    def setup(self):
        with open("secret.json") as f:
            huggingface_token = json.load(f)["HUGGINGFACE_TOKEN"]
        huggingface_hub.login(huggingface_token)

        device = "cuda:0"
        self.if_I = IFStageI("IF-I-XL-v1.0", device=device)
        self.if_II = IFStageII("IF-II-L-v1.0", device=device)
        self.if_III = StableStageIII("stable-diffusion-x4-upscaler", device=device)
        self.t5 = T5Embedder(device=device)

    def predict(
        self,
        prompt: str = Input(description="Input prompt", default="A painting of a cat"),
        negative_prompt: str = "",
        style_prompt: str = "",
        num_outputs: int = Input(
            description="Number of images to output.",
            ge=1,
            le=5,
            default=1,
        ),
        seed: int = Input(
            description="Random seed. Leave 0 to randomize the seed",
            default=0,
            ge=0,
            le=2**32 - 1,
        ),
        aug_level: float = Input(
            description="Adds additional augmentation to generate more realistic images", default=0.25, ge=0.0, le=1.0
        ),
        guidance_scale: float = Input(
            description="Guidance scale", default=7.0, ge=0.0, le=10.0
        ),
        aspect_ratio: str = Input(
            description="Aspect ratio of the output image", default="1:1"
        ),
        stage3_upscale: bool = Input(
            description="Use 1024x1024 upscaler", default=False
        ),
    ) -> List[Path]:
        paths = []
        seed = random.randint(0, 2**32 - 1) if seed == 0 else seed
        result = dream(
            t5=self.t5,
            if_I=self.if_I,
            if_II=self.if_II,
            if_III=self.if_III,
            disable_watermark=True,
            negative_prompt=negative_prompt,
            style_prompt=style_prompt,
            aug_level=aug_level,
            aspect_ratio=aspect_ratio,
            prompt=[prompt] * num_outputs,
            seed=seed,
            if_I_kwargs={
                "guidance_scale": guidance_scale,
                "sample_timestep_respacing": "smart100",
            },
            if_II_kwargs={
                "guidance_scale": 4.0,
                "sample_timestep_respacing": "smart50",
            },
            if_III_kwargs={
                "guidance_scale": 9.0,
                "noise_level": 20,
                "sample_timestep_respacing": "75",
            },
        )
        for n, image in enumerate(result["III"]):
            image.save(f"/tmp/out-{n}.png")
            paths.append(Path(f"/tmp/out-{n}.png"))
        return paths
