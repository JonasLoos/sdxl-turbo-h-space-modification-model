from cog import BasePredictor, Path, Input
import torch
from diffusers import AutoPipelineForText2Image
import tempfile
    

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.pipe = AutoPipelineForText2Image.from_pretrained('stabilityai/sdxl-turbo', torch_dtype=torch.float16, variant="fp16", cache_dir='model').to('cuda')
        # self.pipe = AutoPipelineForText2Image.from_pretrained('stabilityai/sdxl-turbo', torch_dtype=torch.float16, variant="fp16", cache_dir='model', local_files_only=True).to('cuda')
        self.directions = torch.load('directions.pt').to('cuda')

    def predict(self,
            prompt: str = Input(description="Prompt to generate image for"),
            scales: str = Input(description="Comma separated list of direction modification scales"),
            diffusion_steps: int = Input(description="Number of diffusion steps",  default=1, ge=1),
            seed: int = Input(description="Seed for random number generator", default=42),
    ) -> Path:
        """Run a single prediction on the model"""
        # check inputs
        if diffusion_steps != 1:
            raise NotImplementedError("Diffusion steps != 1 is not supported yet")
        scales = torch.tensor([float(s) for s in scales.split(',')], device='cuda')
        if len(scales) != len(self.directions):
            raise ValueError("Number of scales must match number of directions")

        # run prediction
        def hook_fn(module, input, output):
            '''Modify h-space'''
            return output + (self.directions * scales.reshape((-1,1,1,1))).sum(dim=0, keepdim=True)
        with torch.no_grad(), self.pipe.unet.mid_block.register_forward_hook(hook_fn):
            img = self.pipe(prompt = prompt, num_inference_steps = diffusion_steps, guidance_scale = 0.0, generator = torch.Generator('cuda').manual_seed(seed)).images[0]
        
        # save and return image
        output_path = Path(tempfile.mkdtemp()) / 'output.png'
        img.save(output_path)
        return output_path

