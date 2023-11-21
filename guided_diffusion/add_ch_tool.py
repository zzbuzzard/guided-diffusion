import torch
from torch import nn
import argparse
import unet
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from guided_diffusion.nn import conv_nd


def add_tool_args(parser):
    # parser.add_argument("-m", "--model-path", type=str, help="Model path", required=True)
    parser.add_argument("-o", "--output-path", type=str, help="Output model path", required=True)
    parser.add_argument("-n", "--nch", type=int, help="New input channels", required=True)
    return parser


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        model_path="",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


def change_num_input_channels(model: unet.UNetModel, new_ch: int):
    old_conv = model.input_blocks[0][0]
    out_ch = old_conv.out_channels
    old_ch = old_conv.in_channels

    new_layer = unet.TimestepEmbedSequential(conv_nd(2, new_ch, out_ch, 3, padding=1))
    model.input_blocks[0] = new_layer
    model.in_channels = new_ch

    # Initialise weights to 0 for the new channels
    # Note: assumes new_ch > old_ch
    # Note: assumes the new channels will be placed at the end of the input
    with torch.no_grad():
        new_layer[0].weight[:, :old_ch] = old_conv.weight
        new_layer[0].weight[:, old_ch:] = 0

    return model


if __name__ == "__main__":
    args = add_tool_args(create_argparser()).parse_args()
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys()),
        in_channels=3
    )
    model.load_state_dict(
        torch.load(args.model_path, map_location="cpu")
        #dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model = change_num_input_channels(model, new_ch=args.nch)
    torch.save(model.state_dict(), args.output_path)


