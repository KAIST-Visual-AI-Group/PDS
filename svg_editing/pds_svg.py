import pydiffvg
import argparse

from tqdm import tqdm
import argparse

import argparse
from pathlib import Path

import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image

from pds.utils.trainutil import save_config
from pds.utils.sysutil import clean_gpu
from pds.utils.imageutil import resize
from pds.utils.svgutil import render, clip_curve_shape

from pds.pds import PDSConfig, PDS


style_to_config = {
    'iconography' : {
        'optim_point'       : True,
        'optim_color'       : True,
        'prompt_modifier'   : '. minimal flat 2d vector image. lineal color. trending on artstation'
    },
    
    'pixelart' : {
        'optim_point'       : False,
        'optim_color'       : True,
        'prompt_modifier'   : '. pixel art. trending on artstation'
    },
}

def svg_optimizer(shapes:list, shape_groups:list, optim_point:bool, optim_color:bool, point_lr:float, color_lr:float):
    if optim_point:
        points_vars = []
        for path in shapes:
            path.points.requires_grad = True
            points_vars.append(path.points)
        points_optim = torch.optim.Adam(points_vars, lr=point_lr)

    if optim_color:
        color_vars = {}
        for group in shape_groups:
            group.fill_color.requires_grad = True
            color_vars[group.fill_color.data_ptr()] = group.fill_color
        color_vars = list(color_vars.values())
        color_optim = torch.optim.Adam(color_vars, lr=color_lr)
        
    return points_optim, color_optim

def main(args):
    style_config = style_to_config[args.style]
    
    prompt_modifier = style_config['prompt_modifier']
    optim_point = style_config['optim_point']
    optim_color = style_config['optim_color']
    
    src_prompt = args.src_prompt + prompt_modifier
    tgt_prompt = args.tgt_prompt + prompt_modifier

    pds_config_dict = {
        'sd_pretrained_model_or_path' : 'runwayml/stable-diffusion-v1-5',
        'num_inference_steps' : args.num_inference_steps,
        'min_step_ratio' : args.min_step_ratio,
        'max_step_ratio' : args.max_step_ratio,
        'src_prompt' : src_prompt,
        'tgt_prompt' : tgt_prompt,
        'guidance_scale' : args.guidance_scale,
        'device' : args.device
    }
    pds_config = PDSConfig(**pds_config_dict)
    
    pds = PDS(config = pds_config)
    
    config_dict = vars(args)
    config_dict.update(pds_config_dict)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    save_config(
        f"{args.save_path}/config.yaml", config_dict
    )

    # Initialize SVG scene
    canvas_width, canvas_height, shapes, shape_groups = \
        pydiffvg.svg_to_scene(args.svg_path)

    img = render(canvas_width, canvas_height, shapes, shape_groups, args.save_path, 'init')

    points_optim, color_optim = svg_optimizer(
        shapes,
        shape_groups,
        optim_point,
        optim_color,
        args.point_lr,
        args.color_lr
    )

    img = img[:, :, :3]
    img = img.unsqueeze(0) # HWC -> NCHW
    img = img.permute(0, 3, 1, 2) # NHWC -> NCHW
    img = resize(img)

    # Image to latent (w0)
    with torch.no_grad():
        src_w0 = pds.encode_image(img)
    
    loss_args = {
        'tgt_x0' : None,
        'src_x0' : src_w0
    }
    
    for t in tqdm(range(args.epochs)):
        clean_gpu()

        if optim_point:
            points_optim.zero_grad()
        if optim_color:
            color_optim.zero_grad()

        # Target SVG
        img = render(canvas_width, canvas_height, shapes, shape_groups, args.save_path, f'iter_{t}')
        img = img[:, :, :3]
        img = img.unsqueeze(0) # HWC -> NCHW
        img = img.permute(0, 3, 1, 2) # NHWC -> NCHW
        img = resize(img)
        
        w0 = pds.encode_image(img)

        loss_args['tgt_x0'] = w0

        loss = pds(**loss_args)

        loss.backward()
    
        if optim_point:
            points_optim.step()
        if optim_color:
            color_optim.step()

        clip_curve_shape(shape_groups)

        if t % 10 == 0 or t == args.epochs - 1:
            pydiffvg.save_svg(f'{args.save_path}/iter_{t}.svg',
                            canvas_width, canvas_height, shapes, shape_groups)
    
    # Final result
    img = render(canvas_width, canvas_height, shapes, shape_groups, args.save_path, f'final')

    # Output a video with the results of all the intermediate steps
    if args.out_video:
        from subprocess import call
        call(["ffmpeg", "-framerate", "24", "-i",
            f"{args.save_path}/iter_%d.png", "-vb", "20M",
            f"{args.save_path}/out.mp4"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    boolean = argparse.BooleanOptionalAction
    
    # Prompts    
    parser.add_argument("--src_prompt",             type=str,       required=True)
    parser.add_argument("--tgt_prompt",             type=str,       required=True)
    
    # PDS hyperparamgeters
    parser.add_argument("--min_step_ratio",         type=float,     default=0.1)
    parser.add_argument("--max_step_ratio",         type=float,     default=0.98)
    parser.add_argument("--num_inference_steps",    type=int,       default=500)
    parser.add_argument("--guidance_scale",         type=int,       default=100)
    
    # SVG hyperparameters
    parser.add_argument("--point_lr",               type=float,     default=0.8)
    parser.add_argument("--color_lr",               type=float,     default=0.01)
    parser.add_argument("--style",                  type=str,       default='iconography',  choices=['iconography', 'pixelart'])

    # Paths
    parser.add_argument("--svg_path",               type=str,       required=True)
    parser.add_argument("--save_path",              type=str,       default='./results')

    # Others
    parser.add_argument("--epochs",                 type=int,       default=1000)
    parser.add_argument("--device",                 type=int,       default=torch.device("cuda"))
    parser.add_argument("--out_video",              action=boolean, default=True)
    
    args = parser.parse_args()
    main(args)
