#   small modification to allow negative image embeds


# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from math import ceil
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import PIL
import torch
from transformers import CLIPImageProcessor, CLIPTextModelWithProjection, CLIPTokenizer, CLIPVisionModelWithProjection

from diffusers.models import StableCascadeUNet
from diffusers.schedulers import DDPMWuerstchenScheduler
from diffusers.utils import BaseOutput, logging, replace_example_docstring
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from diffusers.pipelines.wuerstchen.modeling_paella_vq_model import PaellaVQModel


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

DEFAULT_STAGE_C_TIMESTEPS = list(np.linspace(1.0, 2 / 3, 20)) + list(np.linspace(2 / 3, 0.0, 11))[1:]

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import StableCascadePriorPipeline

        >>> prior_pipe = StableCascadePriorPipeline.from_pretrained(
        ...     "stabilityai/stable-cascade-prior", torch_dtype=torch.bfloat16
        ... ).to("cuda")

        >>> prompt = "an image of a shiba inu, donning a spacesuit and helmet"
        >>> prior_output = pipe(prompt)
        ```
"""


@dataclass
class StableCascadePriorPipelineOutput(BaseOutput):
    """
    Output class for WuerstchenPriorPipeline.

    Args:
        image_embeddings (`torch.Tensor` or `np.ndarray`)
            Prior image embeddings for text prompt
        prompt_embeds (`torch.Tensor`):
            Text embeddings for the prompt.
        negative_prompt_embeds (`torch.Tensor`):
            Text embeddings for the negative prompt.
    """

    image_embeddings: Union[torch.Tensor, np.ndarray]
    prompt_embeds: Union[torch.Tensor, np.ndarray]
    prompt_embeds_pooled: Union[torch.Tensor, np.ndarray]
    negative_prompt_embeds: Union[torch.Tensor, np.ndarray]
    negative_prompt_embeds_pooled: Union[torch.Tensor, np.ndarray]


class StableCascadePriorPipeline_DoE(DiffusionPipeline):
    """
    Pipeline for generating image prior for Stable Cascade.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        prior ([`StableCascadeUNet`]):
            The Stable Cascade prior to approximate the image embedding from the text and/or image embedding.
        text_encoder ([`CLIPTextModelWithProjection`]):
            Frozen text-encoder
            ([laion/CLIP-ViT-bigG-14-laion2B-39B-b160k](https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k)).
        feature_extractor ([`~transformers.CLIPImageProcessor`]):
            Model that extracts features from generated images to be used as inputs for the `image_encoder`.
        image_encoder ([`CLIPVisionModelWithProjection`]):
            Frozen CLIP image-encoder ([clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)).
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        scheduler ([`DDPMWuerstchenScheduler`]):
            A scheduler to be used in combination with `prior` to generate image embedding.
        resolution_multiple ('float', *optional*, defaults to 42.67):
            Default resolution for multiple images generated.
    """

    unet_name = "prior"
    text_encoder_name = "text_encoder"
    model_cpu_offload_seq = "image_encoder->text_encoder->prior"
    _optional_components = ["image_encoder", "feature_extractor"]
    _callback_tensor_inputs = ["latents", "text_encoder_hidden_states", "negative_prompt_embeds"]

    def __init__(
        self,
        tokenizer: CLIPTokenizer,
        text_encoder: CLIPTextModelWithProjection,
        prior: StableCascadeUNet,
        scheduler: DDPMWuerstchenScheduler,
        resolution_multiple: float = 42.67,
        feature_extractor: Optional[CLIPImageProcessor] = None,
        image_encoder: Optional[CLIPVisionModelWithProjection] = None,
    ) -> None:
        super().__init__()
        self.register_modules(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            image_encoder=image_encoder,
            feature_extractor=feature_extractor,
            prior=prior,
            scheduler=scheduler,
        )
        self.register_to_config(resolution_multiple=resolution_multiple)

    def prepare_latents(
        self, height, width, num_images_per_prompt, dtype, device, generator, latents, scheduler
    ):
        latent_shape = (
            num_images_per_prompt,
            self.prior.config.in_channels,
            ceil(height / self.config.resolution_multiple),
            ceil(width / self.config.resolution_multiple),
        )

        if latents is None:
            latents = randn_tensor(latent_shape, generator=generator, device=device, dtype=dtype)
        else:
            if latents.shape != latent_shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {latent_shape}")
            latents = latents.to(device)

        latents = latents * scheduler.init_noise_sigma
        return latents

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1

    @property
    def num_timesteps(self):
        return self._num_timesteps

    def get_timestep_ratio_conditioning(self, t, alphas_cumprod):
        s = torch.tensor([0.003])
        clamp_range = [0, 1]
        min_var = torch.cos(s / (1 + s) * torch.pi * 0.5) ** 2
        var = alphas_cumprod[t]
        var = var.clamp(*clamp_range)
        s, min_var = s.to(var.device), min_var.to(var.device)
        ratio = (((var * min_var) ** 0.5).acos() / (torch.pi * 0.5)) * (1 + s) - s
        return ratio

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,

        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 20,
        timesteps: List[float] = None,
        guidance_scale: float = 4.0,
        prompt_embeds: Optional[torch.Tensor] = None,
        prompt_embeds_pooled: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds_pooled: Optional[torch.Tensor] = None,
        image_embeds: Optional[torch.Tensor] = None,
        negative_image_embeds: Optional[torch.Tensor] = None,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pt",
        return_dict: bool = True,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    ):
        """
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide the image generation.
            height (`int`, *optional*, defaults to 1024):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to 1024):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 60):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 8.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `decoder_guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting
                `decoder_guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely
                linked to the text `prompt`, usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
                if `decoder_guidance_scale` is less than `1`).
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            prompt_embeds_pooled (`torch.Tensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            negative_prompt_embeds_pooled (`torch.Tensor`, *optional*):
                Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds_pooled will be generated from `negative_prompt`
                input argument.
            image_embeds (`torch.Tensor`, *optional*):
                Pre-generated image embeddings. Can be used to easily tweak image inputs, *e.g.* prompt weighting. If
                not provided, image embeddings will be generated from `image` input argument if existing.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between: `"pil"` (`PIL.Image.Image`), `"np"`
                (`np.array`) or `"pt"` (`torch.Tensor`).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.ImagePipelineOutput`] instead of a plain tuple.
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.

        Examples:

        Returns:
            [`StableCascadePriorPipelineOutput`] or `tuple` [`StableCascadePriorPipelineOutput`] if `return_dict` is
            True, otherwise a `tuple`. When returning a tuple, the first element is a list with the generated image
            embeddings.
        """

        # 0. Define commonly used variables
        device = self._execution_device
        dtype = next(self.prior.parameters()).dtype
        self._guidance_scale = guidance_scale

        # 2. caption + images
        image_embeds_pooled = image_embeds.repeat(num_images_per_prompt, 1, 1)
#        uncond_image_embeds_pooled = torch.zeros_like(image_embeds_pooled)

        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if self.do_classifier_free_guidance:
            uncond_image_embeds_pooled = negative_image_embeds.repeat(num_images_per_prompt, 1, 1)
            image_embeds = torch.cat([image_embeds_pooled, uncond_image_embeds_pooled], dim=0)
            text_encoder_hidden_states = torch.cat([prompt_embeds, negative_prompt_embeds])
            text_encoder_pooled = torch.cat([prompt_embeds_pooled, negative_prompt_embeds_pooled])
        else:
            image_embeds = image_embeds_pooled
            text_encoder_hidden_states = prompt_embeds
            text_encoder_pooled = prompt_embeds_pooled

        # 4. Prepare and set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latents
        latents = self.prepare_latents(
            height, width, num_images_per_prompt, dtype, device, generator, latents, self.scheduler
        )

        if isinstance(self.scheduler, DDPMWuerstchenScheduler):
            timesteps = timesteps[:-1]

        # 6. Run denoising loop
        if hasattr(self.scheduler, "betas"):
            alphas = 1.0 - self.scheduler.betas
            alphas_cumprod = torch.cumprod(alphas, dim=0)
        else:
            alphas_cumprod = []

        self._num_timesteps = len(timesteps)
        for i, t in enumerate(self.progress_bar(timesteps)):
            if not isinstance(self.scheduler, DDPMWuerstchenScheduler):
                if len(alphas_cumprod) > 0:
                    timestep_ratio = self.get_timestep_ratio_conditioning(t.long().cpu(), alphas_cumprod)
                    timestep_ratio = timestep_ratio.expand(latents.size(0)).to(dtype).to(device)
                else:
                    timestep_ratio = t.float().div(self.scheduler.timesteps[-1]).expand(latents.size(0)).to(dtype)
            else:
                timestep_ratio = t.expand(latents.size(0)).to(dtype)

            # 7. Denoise image embeddings
            predicted_image_embedding = self.prior(
                sample=torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents,
                timestep_ratio=torch.cat([timestep_ratio] * 2) if self.do_classifier_free_guidance else timestep_ratio,
                clip_text_pooled=text_encoder_pooled,
                clip_text=text_encoder_hidden_states,
                clip_img=image_embeds,
                return_dict=False,
            )[0]

            # 8. Check for classifier free guidance and apply it
            if self.do_classifier_free_guidance:
                predicted_image_embedding_text, predicted_image_embedding_uncond = predicted_image_embedding.chunk(2)
                predicted_image_embedding = torch.lerp(
                    predicted_image_embedding_uncond, predicted_image_embedding_text, self.guidance_scale
                )

            # 9. Renoise latents to next timestep
            if not isinstance(self.scheduler, DDPMWuerstchenScheduler):
                timestep_ratio = t
            latents = self.scheduler.step(
                model_output=predicted_image_embedding, timestep=timestep_ratio, sample=latents, generator=generator
            ).prev_sample

            if callback_on_step_end is not None:
                callback_kwargs = {}
                for k in callback_on_step_end_tensor_inputs:
                    callback_kwargs[k] = locals()[k]
                callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                latents = callback_outputs.pop("latents", latents)
                prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

        # Offload all models
        self.maybe_free_model_hooks()

        if output_type == "np":
            latents = latents.cpu().float().numpy()  # float() as bfloat16-> numpy doesnt work
            prompt_embeds = prompt_embeds.cpu().float().numpy()  # float() as bfloat16-> numpy doesnt work
            negative_prompt_embeds = (
                negative_prompt_embeds.cpu().float().numpy() if negative_prompt_embeds is not None else None
            )  # float() as bfloat16-> numpy doesnt work

        if not return_dict:
            return (
                latents,
                prompt_embeds,
                prompt_embeds_pooled,
                negative_prompt_embeds,
                negative_prompt_embeds_pooled,
            )

        return StableCascadePriorPipelineOutput(
            image_embeddings=latents,
            prompt_embeds=prompt_embeds,
            prompt_embeds_pooled=prompt_embeds_pooled,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_prompt_embeds_pooled=negative_prompt_embeds_pooled,
        )


class StableCascadeDecoderPipeline_DoE(DiffusionPipeline):
    unet_name = "decoder"
    model_cpu_offload_seq = "decoder->vqgan"
    _callback_tensor_inputs = [
        "latents",
        "prompt_embeds_pooled",
        "negative_prompt_embeds",
        "image_embeddings",
    ]

    def __init__(
        self,
        decoder: StableCascadeUNet,
        scheduler: DDPMWuerstchenScheduler,
        vqgan: PaellaVQModel,
        latent_dim_scale: float = 10.67,
    ) -> None:
        super().__init__()
        self.register_modules(
            decoder=decoder,
            scheduler=scheduler,
            vqgan=vqgan,
        )
        self.register_to_config(latent_dim_scale=latent_dim_scale)

    def prepare_latents(
        self, image_embeddings, num_images_per_prompt, dtype, device, generator, latents, scheduler
    ):
        _, channels, height, width = image_embeddings.shape
        latents_shape = (
            num_images_per_prompt,
            4,
            int(height * self.config.latent_dim_scale),
            int(width * self.config.latent_dim_scale),
        )

        if latents is None:
            latents = randn_tensor(latents_shape, generator=generator, device=device, dtype=dtype)
        else:
            if latents.shape != latents_shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {latents_shape}")
            latents = latents.to(device)

        latents = latents * scheduler.init_noise_sigma
        return latents

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1

    @property
    def num_timesteps(self):
        return self._num_timesteps

    def get_timestep_ratio_conditioning(self, t, alphas_cumprod):
        s = torch.tensor([0.003])
        clamp_range = [0, 1]
        min_var = torch.cos(s / (1 + s) * torch.pi * 0.5) ** 2
        var = alphas_cumprod[t]
        var = var.clamp(*clamp_range)
        s, min_var = s.to(var.device), min_var.to(var.device)
        ratio = (((var * min_var) ** 0.5).acos() / (torch.pi * 0.5)) * (1 + s) - s
        return ratio

    @torch.no_grad()
    def __call__(
        self,
        image_embeddings: Union[torch.Tensor, List[torch.Tensor]],
        prompt: Union[str, List[str]] = None,
        num_inference_steps: int = 10,
        guidance_scale: float = 0.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        prompt_embeds_pooled: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds_pooled: Optional[torch.Tensor] = None,
        num_images_per_prompt: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    ):

        # 0. Define commonly used variables
        device = self._execution_device
        dtype = self.decoder.dtype
        self._guidance_scale = guidance_scale

        # 1. Check inputs. Raise error if not correct
        if isinstance(image_embeddings, list):
            image_embeddings = torch.cat(image_embeddings, dim=0)

        # 2. Encode caption
        # The pooled embeds from the prior are pooled again before being passed to the decoder
        prompt_embeds_pooled = (
            torch.cat([prompt_embeds_pooled, negative_prompt_embeds_pooled])
            if self.do_classifier_free_guidance
            else prompt_embeds_pooled
        )
        effnet = (
            torch.cat([image_embeddings, torch.zeros_like(image_embeddings)])
            if self.do_classifier_free_guidance
            else image_embeddings
        )

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latents
        latents = self.prepare_latents(
            image_embeddings, num_images_per_prompt, dtype, device, generator, latents, self.scheduler
        )

        if isinstance(self.scheduler, DDPMWuerstchenScheduler):
            timesteps = timesteps[:-1]

        # 6. Run denoising loop
        if hasattr(self.scheduler, "betas"):
            alphas = 1.0 - self.scheduler.betas
            alphas_cumprod = torch.cumprod(alphas, dim=0)
        else:
            alphas_cumprod = []

        self._num_timesteps = len(timesteps)
        for i, t in enumerate(self.progress_bar(timesteps)):
            if not isinstance(self.scheduler, DDPMWuerstchenScheduler):
                if len(alphas_cumprod) > 0:
                    timestep_ratio = self.get_timestep_ratio_conditioning(t.long().cpu(), alphas_cumprod)
                    timestep_ratio = timestep_ratio.expand(latents.size(0)).to(dtype).to(device)
                else:
                    timestep_ratio = t.float().div(self.scheduler.timesteps[-1]).expand(latents.size(0)).to(dtype)
            else:
                timestep_ratio = t.expand(latents.size(0)).to(dtype)

            # 7. Denoise latents
            predicted_latents = self.decoder(
                sample=torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents,
                timestep_ratio=torch.cat([timestep_ratio] * 2) if self.do_classifier_free_guidance else timestep_ratio,
                clip_text_pooled=prompt_embeds_pooled,
                effnet=effnet,
                return_dict=False,
            )[0]

            # 8. Check for classifier free guidance and apply it
            if self.do_classifier_free_guidance:
                predicted_latents_text, predicted_latents_uncond = predicted_latents.chunk(2)
                predicted_latents = torch.lerp(predicted_latents_uncond, predicted_latents_text, self.guidance_scale)

            # 9. Renoise latents to next timestep
            if not isinstance(self.scheduler, DDPMWuerstchenScheduler):
                timestep_ratio = t
            latents = self.scheduler.step(
                model_output=predicted_latents,
                timestep=timestep_ratio,
                sample=latents,
                generator=generator,
            ).prev_sample

            if callback_on_step_end is not None:
                callback_kwargs = {}
                for k in callback_on_step_end_tensor_inputs:
                    callback_kwargs[k] = locals()[k]
                callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                latents = callback_outputs.pop("latents", latents)
                prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

        if output_type not in ["pt", "np", "pil", "latent"]:
            raise ValueError(
                f"Only the output types `pt`, `np`, `pil` and `latent` are supported not output_type={output_type}"
            )

        if not output_type == "latent":
            # 10. Scale and decode the image latents with vq-vae
            latents = self.vqgan.config.scale_factor * latents
            images = self.vqgan.decode(latents).sample.clamp(0, 1)
            if output_type == "np":
                images = images.permute(0, 2, 3, 1).cpu().float().numpy()  # float() as bfloat16-> numpy doesnt work
            elif output_type == "pil":
                images = images.permute(0, 2, 3, 1).cpu().float().numpy()  # float() as bfloat16-> numpy doesnt work
                images = self.numpy_to_pil(images)
        else:
            images = latents

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return images
        return ImagePipelineOutput(images)
