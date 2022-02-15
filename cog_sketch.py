# Prediction interface for Cog ⚙️
# Reference: https://github.com/replicate/cog/blob/main/docs/python.md

import cog

import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

from tqdm import tqdm
from torchvision import transforms, models

import pydiffvg

import torch
import math
import torch.nn.functional as F
import torch.nn as nn
import os
import sketch_utils as utils
import time
import config
from dataclasses import dataclass
import numpy as np
import random
from pathlib import Path
import PIL
from PIL import Image
from models.loss import Loss
from models.painter_params import Painter, PainterOptimizer


@dataclass
class LossConfig:
    percep_loss: str = None
    train_with_clip: float = 0
    clip_weight: float = 0
    start_clip: int = 0
    clip_conv_loss: float = 1.0
    clip_fc_loss_weight: float = 0.1
    clip_text_guide: float = 0.0
    device: str = 'cuda:0'
    clip_conv_layer_weights: str = "0,0,1.0,1.0,0"
    num_aug_clip: int = 4
    include_target_in_aug: int = 0
    augment_both: int = 1
    augmentations: str = 'affine'


@dataclass
class PainterConfig:
    control_points_per_seg: int
    force_sparse: float
    num_stages: int
    augmentations: str
    noise_thresh: float
    softmax_temp: float
    color_vars_threshold: float
    attention_init: int
    target: Path
    mask_object_attention: int
    text_target: str
    num_iter: int
    xdog_intersec: int = 1
    path_svg: str = 'none'
    saliency_model: str = 'clip'
    saliency_clip_model: str = 'ViT-B/32'


@dataclass
class OptimizerConfig:
    lr: float
    color_lr: float
    force_sparse: float


@dataclass
class TargetConfig:
    device: str = 'cuda:0'
    output_dir: str = 'temp'
    use_gpu: bool = True


def load_renderer(args, num_paths, num_segments, image_scale, device, target_im=None, mask=None):
    renderer = Painter(
        num_strokes=num_paths,
        args=args,
        num_segments=num_segments,
        imsize=image_scale,
        device=device,
        target_im=target_im,
        mask=mask
    )
    renderer = renderer.to(device)
    return renderer


def get_target(target, mask_object, fix_scale, image_scale, device):
    target = Image.open(target)
    if target.mode == "RGBA":
        # Create a white rgba background
        new_image = Image.new("RGBA", target.size, "WHITE")
        # Paste the image on the background.
        new_image.paste(target, (0, 0), target)
        target = new_image
    target = target.convert("RGB")
    masked_im, mask = utils.get_mask_u2net(args, target)
    if mask_object:
        target = masked_im
    if fix_scale:
        target = utils.fix_image_scale(target)

    transforms_ = []
    if target.size[0] != target.size[1]:
        transforms_.append(transforms.Resize(
            (image_scale, image_scale), interpolation=PIL.Image.BICUBIC))
    else:
        transforms_.append(transforms.Resize(
            image_scale, interpolation=PIL.Image.BICUBIC))
        transforms_.append(transforms.CenterCrop(image_scale))
    transforms_.append(transforms.ToTensor())
    data_transforms = transforms.Compose(transforms_)
    target_ = data_transforms(target).unsqueeze(0).to(device)
    return target_, mask



class LossWrapper(Loss):

    @classmethod
    def from_args(
        cls,
        percep_loss,
        train_with_clip,
        clip_weight,
        start_clip,
        clip_conv_loss,
        clip_fc_loss_weight, 
        clip_text_guide,
        device,
        clip_conv_layer_weights,
        num_aug_clip
    ):
        config = LossConfig(percep_loss, train_with_clip, clip_weight, start_clip, clip_conv_loss, 
            clip_fc_loss_weight, clip_text_guide, device, clip_conv_layer_weights, num_aug_clip)
        return cls(config)


class Predictor(cog.Predictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        assert torch.cuda.is_available() and torch.cuda.device_count() > 0
        self.device = torch.device('cuda:0')
        pydiffvg.set_use_gpu(True)
        pydiffvg.set_device(self.device)
    
    def set_seed(self, seed: int = 0):
        random.seed(seed)
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
       


    # =================================
    # ============ general ============
    # =================================
    @cog.input("target", type=Path, help="target image path")
    @cog.input("seed", type=int, default=0)
    @cog.input("mask_object", type=int, default=0)
    @cog.input("fix_scale", type=int, default=0)
    # =================================
    # =========== training ============
    # =================================
    @cog.input(
        "num_iter",
        type=int,
        min=0,
        max=750,
        default=500,
        help="number of optimization iterations")
    @cog.input(
        "num_stages",
        type=int,
        default=1,
        help="training stages, you can train x strokes, then freeze them and train another x strokes etc.")
    @cog.input("lr", type=float, default=1.0)
    @cog.input("color_lr", type=float, default=0.01)
    @cog.input("color_vars_threshold", type=float, default=0.0)
    @cog.input("eval_interval", type=int, default=10)
    @cog.input("image_scale", type=int, default=224)
    # =================================
    # ======== strokes params =========
    # =================================
    @cog.input(
        "num_paths",
        type=int,
        default=16,
        help="number of strokes")
    # @cog.input(
    #    "width",
    #    type=float,
    #    default=1.5,
    #    help="stroke width")
    @cog.input("control_points_per_seg", type=int, default=4)
    @cog.input(
        "num_segments",
        type=int,
        default=1,
        help="number of segments for each stroke, each stroke is a bezier curve with 4 control points")
    @cog.input(
        "attention_init",
        type=int,
        default=1,
        help="if True, use the attention heads of Dino model to set the location of the initial strokes")
    @cog.input("xdog_intersec", type=int, default=1)
    @cog.input("mask_object_attention", type=int, default=0)
    @cog.input("softmax_temp", type=float, default=0.3)
    # =================================
    # ============= loss ==============
    # =================================
    @cog.input(
        "percep_loss",
        type=str,
        default="none",
        options=['none', 'L2', 'LPIPS'],
        help="the type of perceptual loss to be used (L2/LPIPS/none)")
    @cog.input("train_with_clip", type=int, default=0)
    @cog.input("clip_weight", type=float, default=0)
    @cog.input("start_clip", type=int, default=0)
    @cog.input("num_aug_clip", type=int, default=4)
    @cog.input("include_target_in_aug", type=int, default=0)
    @cog.input(
        "augment_both",
        type=int,
        default=1,
        help="if you want to apply the affine augmentation to both the sketch and image")
    @cog.input(
        "augmentations",
        type=str,
        default="affine",
        help="can be any combination of: 'affine_noise_eraserchunks_eraser_press'")
    @cog.input("noise_thresh", type=float, default=0.5)
    # @cog.input("aug_scale_min", type=float, default=0.7)
    @cog.input(
        "force_sparse",
        type=float,
        default=0,
        help="if True, use L1 regularization on stroke's opacity to encourage small number of strokes")
    @cog.input("clip_conv_loss", type=float, default=1)
    @cog.input("clip_conv_loss_type", type=str, default="L2", options=["L2", "Cos", "L1"])
    @cog.input("clip_conv_layer_weights",
                        type=str, default="0,0,1.0,1.0,0")
    @cog.input("clip_fc_loss_weight", type=float, default=0.1)
    @cog.input("clip_text_guide", type=float, default=0)
    @cog.input("text_target", type=str, default="none")
    def predict(
        self,
        target,
        seed,
        mask_object,
        fix_scale,
        num_iter,
        num_stages,
        lr,
        color_lr,
        color_vars_threshold,
        eval_interval,
        image_scale,
        num_paths,
        # width,
        control_points_per_seg,
        num_segments,
        attention_init,
        xdog_intersec,
        mask_object_attention,
        softmax_temp,
        percep_loss,
        train_with_clip,
        clip_weight,
        start_clip,
        num_aug_clip,
        include_target_in_aug,
        aug_both,
        augmentations,
        noise_thresh,
        # aug_scale_min,
        force_sparse,
        clip_conv_loss,
        clip_conv_loss_type,
        clip_conv_layer_weights,
        clip_fc_loss_weight,
        clip_text_guide,
        text_target
    ):
        """Run a single prediction on the model"""
        self.set_seed(seed)
        painter_config = PainterConfig(
            control_points_per_seg,
            force_sparse,
            num_stages,
            augmentations,
            noise_thresh,
            softmax_temp,
            color_vars_threshold,
            attention_init,
            target,
            mask_object_attention,
            text_target,
            num_iter,
            xdog_intersec
        )
        loss_func = LossWrapper.from_args(
            percep_loss,
            train_with_clip,
            clip_weight,
            start_clip,
            clip_conv_loss,
            clip_fc_loss_weight, 
            clip_text_guide,
            self.device,
            clip_conv_layer_weights,
            num_aug_clip,
            augmentations
        )
        inputs, mask = get_target(
            TargetConfig(device=self.device),
            target,
            mask_object,
            fix_scale,
            image_scale,
            self.device
        )
        renderer = load_renderer(
            painter_config,
            num_paths,
            num_segments,
            image_scale,
            self.device,
            mask=mask
        )
        optimizer = PainterOptimizer(OptimizerConfig(lr, color_lr, force_sparse), renderer)
        counter = 0
        best_loss, best_fc_loss = 100, 100
        best_iter, best_iter_fc = 0, 0
        min_delta = 1e-5
        terminate = False
        renderer.set_random_noise(0)
        img = renderer.init_image(stage=0)
        optimizer.init_optimizers()
        for epoch in tqdm(range(num_iter)):
            renderer.set_random_noise(epoch)
            optimizer.zero_grad_()
            sketches = renderer.get_image().to(self.device)
            losses_dict = loss_func(sketches, inputs.detach(
            ), renderer.get_color_parameters(), renderer, counter, optimizer)
            loss = sum(list(losses_dict.values()))
            loss.backward()
            optimizer.step_()
            if epoch % eval_interval == 0:
                with torch.no_grad():
                    losses_dict_eval = loss_func(
                        sketches,
                        inputs,
                        renderer.get_color_parameters(),
                        renderer.get_points_parans(),
                        counter,
                        optimizer,
                        mode="eval"
                    )
                    loss_eval = sum(list(losses_dict_eval.values()))
                    if clip_fc_loss_weight:
                        if losses_dict_eval["fc"].item() < best_fc_loss:
                            best_fc_loss = (
                                losses_dict_eval["fc"].item() / clip_fc_loss_weight
                            )
                            best_iter_fc = epoch
                    cur_delta = loss_eval.item() - best_loss
                    if abs(cur_delta) > min_delta:
                        if cur_delta < 0:
                            best_loss = loss_eval.item()
                            best_iter = epoch
                            terminate = False
                            renderer.save_svg("best_iter")
                    if abs(cur_delta) <= min_delta:
                        if terminate:
                            break
                        terminate = True
            counter += 1
        # renderer.save_svg(output_dir, "final_svg")
        path_svg = "best_iter.svg"
        return Path(path_svg)

