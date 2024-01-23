from diffusers import AutoPipelineForText2Image
import torch
from tqdm import tqdm
from functools import cache
import json


pipe = AutoPipelineForText2Image.from_pretrained('stabilityai/sdxl-turbo', torch_dtype=torch.float16, variant="fp16").to('cuda')
pipe.set_progress_bar_config(disable=True)
attributes = json.load(open('attributes.json'))


@cache
def get_representation(text, count):
    '''Get the representation for a given prompt.'''
    reprs = []
    def get_repr(module, input, output):
        # take mean over spatial dimensions
        reprs.append(output.detach())
    for i in range(count):
        with torch.no_grad(), pipe.unet.mid_block.register_forward_hook(get_repr):
            pipe(prompt = text, num_inference_steps = 1, guidance_scale=0.0)
    return torch.cat(reprs).mean(axis=0).cpu()


def save_directions():
    directions = []
    for prompt in tqdm(attributes, desc='Base prompts'):
        base_prompt = prompt['base_prompt']
        base_repr = get_representation(base_prompt, 50)
        for attribute in tqdm(prompt['attributes'], desc='Attributes', leave=False):
            name = attribute['attribute']
            prompt = attribute['prompt']
            prompt_repr = get_representation(prompt, 50)
            directions.append(prompt_repr - base_repr)
    directions = torch.stack(directions)
    print(f'Directions shape: {directions.shape}')
    torch.save(directions, 'directions.pt')


if __name__ == '__main__':
    save_directions()
