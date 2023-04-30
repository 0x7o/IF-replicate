from diffusers import DiffusionPipeline
from diffusers.utils import pt_to_pil
from cog import BasePredictor, File
import huggingface_hub
import random
import torch
import os


class Predictor(BasePredictor):
    def setup(self):
        huggingface_hub.login(os.environ["HUGGINGFACE_TOKEN"])
        self.stage_1 = DiffusionPipeline.from_pretrained(
            "DeepFloyd/IF-I-XL-v1.0", variant="fp16", torch_dtype=torch.float16
        )
        """self.stage_2 = DiffusionPipeline.from_pretrained(
            "DeepFloyd/IF-II-L-v1.0",
            text_encoder=None,
            variant="fp16",
            torch_dtype=torch.float16,
        )
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

    def predict(self, prompt: str = "A beautiful landscape") -> File:
        prompt_embeds, negative_embeds = self.stage_1.encode_prompt(prompt)
        generator = torch.manual_seed(random.randint(0, 2**32 - 1))
        image = self.stage_1(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_embeds,
            generator=generator,
            output_type="pt",
        ).images
        return File(pt_to_pil(image))
