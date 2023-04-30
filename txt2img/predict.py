from cog import BasePredictor, Path, Input
from diffusers import DiffusionPipeline
from diffusers.utils import pt_to_pil
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
        self.stage_1 = DiffusionPipeline.from_pretrained(
            "DeepFloyd/IF-I-XL-v1.0",
            variant="fp16",
            torch_dtype=torch.float16,
            watermarker=None,
        ).to("cuda")
        self.stage_2 = DiffusionPipeline.from_pretrained(
            "DeepFloyd/IF-II-L-v1.0",
            text_encoder=None,
            variant="fp16",
            torch_dtype=torch.float16,
        ).to("cuda")
        """
        safety_modules = {
            "feature_extractor": self.stage_1.feature_extractor,
            "safety_checker": self.stage_1.safety_checker,
            "watermarker": self.stage_1.watermarker,
        }
        self.stage_3 = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-x4-upscaler",
            **safety_modules,
            torch_dtype=torch.float16
        )"""

    def predict(
        self,
        prompt: str = Input(description="Input prompt", default="A painting of a cat"),
        negative_prompt: str = Input(
            description="Specify things to not see in the output", default=None
        ),
        num_inference_steps: int = Input(
            description="Number of inference steps",
            default=50,
            ge=1,
            le=100,
        ),
        num_outputs: int = Input(
            description="Number of images to output.",
            ge=1,
            le=4,
            default=1,
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=0
        ),
        guidance_scale: float = Input(
            description="Guidance scale", default=7.0, ge=0.0, le=10.0
        ),
    ) -> List[Path]:
        prompt_embeds, negative_embeds = self.stage_1.encode_prompt(
            prompt, negative_prompt, device="cuda"
        )
        paths = []
        for n in range(num_outputs):
            seed = random.randint(0, 2**16 - 1) if seed == 0 else seed
            generator = torch.manual_seed(seed)
            image = self.stage_1(
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_embeds,
                generator=generator,
                output_type="pt",
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
            ).images
            image = self.stage_2(
                image=image,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_embeds
                if negative_prompt is not None
                else None,
                do_classifier_free_guidance=True,
                generator=generator,
                output_type="pt",
                num_inference_steps=75,
            ).images
            pt_to_pil(image)[0].save(f"/tmp/out-{n}.png")
            paths.append(Path(f"/tmp/out-{n}.png"))
        return paths
