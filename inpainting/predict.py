from diffusers import (
    IFInpaintingPipeline,
    IFInpaintingSuperResolutionPipeline,
    DiffusionPipeline,
)
from cog import BasePredictor, Path, Input
from diffusers.utils import pt_to_pil
from PIL import Image
from typing import List
import huggingface_hub
import random
import torch
import json


class Predictor(BasePredictor):
    def setup(self):
        with open("secret.json") as f:
            huggingface_token = json.load(f)["HUGGINGFACE_TOKEN"]
        huggingface_hub.login(huggingface_token)
        self.stage_1 = IFInpaintingPipeline.from_pretrained(
            "DeepFloyd/IF-I-XL-v1.0",
            variant="fp16",
            torch_dtype=torch.float16,
            watermarker=None,
        ).to("cuda")
        self.stage_2 = IFInpaintingSuperResolutionPipeline.from_pretrained(
            "DeepFloyd/IF-II-L-v1.0",
            text_encoder=None,
            variant="fp16",
            torch_dtype=torch.float16,
        ).to("cuda")
        safety_modules = {
            "feature_extractor": self.stage_1.feature_extractor,
            "safety_checker": self.stage_1.safety_checker,
            "watermarker": self.stage_1.watermarker,
        }
        self.stage_3 = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-x4-upscaler",
            **safety_modules,
            torch_dtype=torch.float16,
        ).to("cuda")

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
        num_inference_steps: int = Input(
            description="Number of inference steps",
            default=50,
            ge=1,
            le=100,
        ),
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
            description="Guidance scale", default=10.0, ge=0.0, le=15.0
        ),
        stage3_upscale: bool = Input(
            description="Use 1024x1024 upscaler", default=False
        ),
    ) -> List[Path]:
        paths = []
        seed = random.randint(0, 2**32 - 1) if seed == 0 else seed
        generator = torch.Generator(device="cuda").manual_seed(seed)
        prompt_embeds, negative_embeds = self.stage_1.encode_prompt(
            prompt=prompt, negative_prompt=negative_prompt, device="cuda"
        )
        original_image = self.resize_image(Image.open(original_image).convert("RGB"))
        mask_image = self.resize_image(Image.open(mask_image))
        for n in range(num_outputs):
            image = self.stage_1(
                image=original_image,
                mask_image=mask_image,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_embeds,
                generator=generator,
                output_type="pt",
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
            ).images
            image = self.stage_2(
                image=image,
                mask_image=mask_image,
                original_image=original_image,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_embeds,
                guidance_scale=4.0,
                generator=generator,
                output_type="pt",
                num_inference_steps=50,
                noise_level=250,
            ).images
            if stage3_upscale:
                image = self.stage_3(
                    image=image,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_images_per_prompt=1,
                    guidance_scale=9.0,
                    num_inference_steps=75,
                    generator=generator,
                    noise_level=100,
                ).images
                image = image[0]
            else:
                image = pt_to_pil(image)[0]
            image.save(f"/tmp/out-{n}.png")
            paths.append(Path(f"/tmp/out-{n}.png"))
        return paths
