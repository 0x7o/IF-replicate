from deepfloyd_if.modules import IFStageI, IFStageII, StableStageIII
from deepfloyd_if.modules.t5 import T5Embedder
from deepfloyd_if.pipelines import inpainting
from cog import BasePredictor, Path, Input
from PIL import Image
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

    def clamp(self, value, min_value, max_value) -> int:
        return max(min(value, max_value), min_value)

    def resize_image(self, original_image: Image) -> Image:
        width, height = original_image.size

        width = self.clamp((width // 2) * 2, 512, 1024)
        height = self.clamp((height // 2) * 2, 512, 1024)

        resized_image = original_image.resize((width, height), Image.ANTIALIAS)
        return resized_image

    def predict(
        self,
        original_image: Path = Input(
            description="Inital image to generate variations of.",
        ),
        mask_image: Path = Input(
            description="Mask image to use for inpainting.",
        ),
        prompt: str = Input(description="Input prompt", default="blue sunglasses"),
        negative_prompt: str = "",
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
        guidance_scale: float = Input(
            description="Guidance scale", default=7.0, ge=0.0, le=10.0
        ),
    ) -> List[Path]:
        paths = []
        seed = random.randint(0, 2**32 - 1) if seed == 0 else seed
        original_image = self.resize_image(Image.open(original_image).convert("RGB"))
        mask_image = self.resize_image(Image.open(mask_image).convert("RGB"))
        result = inpainting(
            t5=self.t5,
            if_I=self.if_I,
            if_II=self.if_II,
            if_III=self.if_III,
            disable_watermark=True,
            support_pil_img=original_image,
            negative_prompt=negative_prompt,
            inpainting_mask=mask_image,
            prompt=[prompt] * num_outputs,
            seed=seed,
            if_I_kwargs={
                "guidance_scale": guidance_scale,
                "sample_timestep_respacing": "10,10,10,10,10,0,0,0,0,0",
                "support_noise_less_qsample_steps": 0,
            },
            if_II_kwargs={
                "guidance_scale": 4.0,
                "aug_level": 0.0,
                "sample_timestep_respacing": "100",
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
