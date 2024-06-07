import gradio as gr
import torch
import gc
import json
from diffusers import StableCascadeDecoderPipeline, StableCascadePriorPipeline, StableCascadeUNet#, DDPMWuerstchenScheduler

from diffusers import DPMSolverSinglestepScheduler, DPMSolverMultistepScheduler
from diffusers import SASolverScheduler
from diffusers import EulerAncestralDiscreteScheduler, EulerDiscreteScheduler, UniPCMultistepScheduler

from modules import ui_common

from modules import script_callbacks, images, shared
from modules.processing import get_fixed_seed
from modules.rng import create_generator
from modules.shared import opts
from modules.ui_components import ResizeHandleRow
import modules.infotext_utils as parameters_copypaste

torch.backends.cuda.enable_mem_efficient_sdp(True)

import customStylesList as styles

class CascadeMemory:
    lastSeed = -1
    galleryIndex = 0
    karras = False


# modules/infotext_utils.py
def quote(text):
    if ',' not in str(text) and '\n' not in str(text) and ':' not in str(text):
        return text

    return json.dumps(text, ensure_ascii=False)

# modules/processing.py
def create_infotext(priorModel, decoderModel, positive_prompt, negative_prompt, guidance_scale, prior_steps, decoder_steps, seed, scheduler, width, height, ):
    karras = " : Karras" if CascadeMemory.karras == True else ""
    generation_params = {
        "Size": f"{width}x{height}",
        "Seed": seed,
        "Scheduler": f"{scheduler}{karras}",
        "Steps(Prior/Decoder)": f"{prior_steps}/{decoder_steps}",
        "CFG": guidance_scale,
        "RNG": opts.randn_source if opts.randn_source != "GPU" else None,
    }

#add i2i marker?
    model_text = "(" + priorModel.split('.')[0] + "/" + decoderModel.split('.')[0] + ")"
    
    prompt_text = f"Prompt: {positive_prompt}"
    if negative_prompt != "":
        prompt_text += (f"\nNegative: {negative_prompt}")
    generation_params_text = ", ".join([k if k == v else f'{k}: {quote(v)}' for k, v in generation_params.items() if v is not None])

    return f"Model: StableCascade {model_text}\n{prompt_text}\n{generation_params_text}"


def predict(priorModel, decoderModel, positive_prompt, negative_prompt, width, height, guidance_scale,
            prior_steps, decoder_steps, seed, batch_size, PriorScheduler, style, i2iSource, i2iStrength,):
            #resolution, latentScale):

    if style != 0:
        positive_prompt = styles.styles_list[style][1].replace("{prompt}", positive_prompt)
        negative_prompt = styles.styles_list[style][2] + negative_prompt


    gc.collect()
    torch.cuda.empty_cache()

    fixed_seed = get_fixed_seed(seed)
    CascadeMemory.lastSeed = fixed_seed

    useLitePrior = "lite" in priorModel
    useLiteDecoder = "lite" in decoderModel

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() == True else torch.float16

    if i2iSource != None:
        i2iSource = i2iSource.resize([width, height])
        prior = StableCascadePriorPipeline.from_pretrained(
            "stabilityai/stable-cascade-prior", 
            local_files_only=False, cache_dir=".//models//diffusers//",
            prior=None,
            text_encoder=None,
            tokenizer=None,
            scheduler=None,
            variant="bf16",
            torch_dtype=torch.float32)
        with torch.no_grad():
            image_embeds, neg_image_embeds = prior.encode_image(images=[i2iSource], device='cpu', dtype=torch.float32, batch_size=1, num_images_per_prompt=1)
            image_embeds = image_embeds * i2iStrength
            image_embeds = image_embeds.to('cuda').to(dtype)
        del prior
    else:
        image_embeds = None
    
#cache embeds? not slow enough to concern
#process on GPU?

    if priorModel == "lite":
        prior_unet = StableCascadeUNet.from_pretrained("stabilityai/stable-cascade-prior", 
            local_files_only=False, cache_dir=".//models//diffusers//",
            subfolder="prior_lite",
            variant="bf16", torch_dtype=dtype)
    elif priorModel == "full":
        prior_unet = StableCascadeUNet.from_pretrained("stabilityai/stable-cascade-prior", 
            local_files_only=False, cache_dir=".//models//diffusers//",
            subfolder="prior",
            variant="bf16", torch_dtype=dtype)
    else:
        prior_unet = StableCascadeUNet.from_single_file(".//models//diffusers//StableCascadeCustom//StageC//" + priorModel,
            local_files_only=True, cache_dir=".//models//diffusers//StableCascadeCustom//StageC//",
            torch_dtype=dtype)


    #auto calc resolution_multiple based on shortest dimension?
#    resolution = min(width, height) / 24

    prior = StableCascadePriorPipeline.from_pretrained("stabilityai/stable-cascade-prior", 
        local_files_only=False, cache_dir=".//models//diffusers//",
        image_encoder=None, feature_extractor=None,
        prior=prior_unet,
        variant="bf16", torch_dtype=dtype,)
#        resolution_multiple = resolution)

    prior.to('cuda')
    prior.enable_attention_slicing()
    if useLitePrior == False:
        prior.enable_sequential_cpu_offload()  #good for full on 8GB, but slows down lite significantly?

    if PriorScheduler == 'DPM++ 2M':
        prior.scheduler = DPMSolverMultistepScheduler.from_config(prior.scheduler.config)
    elif PriorScheduler == "DPM++ 2M SDE":
        prior.scheduler = DPMSolverMultistepScheduler.from_config(prior.scheduler.config, algorithm_type='sde-dpmsolver++')
    elif PriorScheduler == "SA-solver":
        prior.scheduler = SASolverScheduler.from_config(prior.scheduler.config, algorithm_type='data_prediction')
####   else use default
    
    prior.scheduler.config.use_karras_sigmas = CascadeMemory.karras
    prior.scheduler.config.clip_sample = False

    generator = [torch.Generator().manual_seed(fixed_seed+i) for i in range(batch_size)]

    prior_output = prior(
        prompt=positive_prompt,
        negative_prompt=negative_prompt,

        image_embeds=image_embeds,

        width=width,
        height=height,
        guidance_scale=guidance_scale,
        num_inference_steps=prior_steps,
        num_images_per_prompt=batch_size,
        generator=generator
    )

    prompt_embeds = prior_output.get("prompt_embeds", None)
    prompt_embeds_pooled = prior_output.get("prompt_embeds_pooled", None)
    negative_prompt_embeds = prior_output.get("negative_prompt_embeds", None)
    negative_prompt_embeds_pooled = prior_output.get("negative_prompt_embeds_pooled", None)

    del prior, generator

    gc.collect()
    torch.cuda.empty_cache()

    if decoderModel == "lite":
        decoder_unet = StableCascadeUNet.from_pretrained("stabilityai/stable-cascade", 
            local_files_only=False, cache_dir=".//models//diffusers//",
            subfolder="decoder_lite", variant="bf16", torch_dtype=dtype)
    elif decoderModel == "full":
        decoder_unet = StableCascadeUNet.from_pretrained("stabilityai/stable-cascade", 
            local_files_only=False, cache_dir=".//models//diffusers//",
            subfolder="decoder", variant="bf16", torch_dtype=dtype)
    else:
        decoder_unet = StableCascadeUNet.from_single_file(".//models//diffusers//StableCascadeCustom//StageB//" + decoderModel, 
            local_files_only=True, cache_dir=".//models//diffusers//StableCascadeCustom//StageB//",
            subfolder="decoder", torch_dtype=dtype)

    decoder = StableCascadeDecoderPipeline.from_pretrained("stabilityai/stable-cascade", 
        local_files_only=False, cache_dir=".//models//diffusers//",
        tokenizer=None, text_encoder=None,
        decoder=decoder_unet, variant="bf16", torch_dtype=dtype,)
        #latent_dim_scale = latentScale)

    decoder.to('cuda')
    decoder.enable_attention_slicing()
    if useLiteDecoder == False:
#        decoder.enable_sequential_cpu_offload()  #necessary?
        decoder.enable_model_cpu_offload()

    ##  regenerate the Generator, needed for deterministic outputs - reusing from earlier doesn't work
        #still not correct with custom checkpoint?
    generator = [torch.Generator().manual_seed(fixed_seed+i) for i in range(batch_size)]

    decoder_output = decoder(
        image_embeddings=prior_output.image_embeddings.to(dtype),
        prompt_embeds = prompt_embeds,
        prompt_embeds_pooled = prompt_embeds_pooled,
        negative_prompt_embeds = negative_prompt_embeds,
        negative_prompt_embeds_pooled = negative_prompt_embeds_pooled,
        prompt=None,
        negative_prompt=None,
        guidance_scale=1,
        output_type="pil",
        num_inference_steps=decoder_steps,
        generator=generator,
    ).images

    del prior_output, prompt_embeds, prompt_embeds_pooled, negative_prompt_embeds, negative_prompt_embeds_pooled
    del generator, decoder

    gc.collect()
    torch.cuda.empty_cache()

    result = []

    for image in decoder_output:
        info=create_infotext(priorModel, decoderModel, positive_prompt, negative_prompt, guidance_scale, prior_steps, decoder_steps, fixed_seed,
                             PriorScheduler, width, height)
        result.append((image, info))
        images.save_image(
            image,
            opts.outdir_samples or opts.outdir_txt2img_samples,
            "",
            fixed_seed,
            positive_prompt,
            opts.samples_format,
            info
        )
        fixed_seed += 1

    gc.collect()
    torch.cuda.empty_cache()

    return result, gr.Button.update(value='Generate', variant='primary', interactive=True)


def on_ui_tabs():
    from modules.ui_components import ToolButton                                                     

    styles_list = ["(None)",
                   "Cinematic", "Photographic",
                   "Anime", "Manga",
                   "Digital art", "Pixel art",
                   "Fantasy art", "Neonpunk", "3D model",
                  ]
    def buildModelsLists ():
        prior = ["lite", "full"]
        decoder = ["lite", "full"]
        
        import glob
        customStageC = glob.glob(".\models\diffusers\StableCascadeCustom\StageC\*.safetensors")
        customStageB = glob.glob(".\models\diffusers\StableCascadeCustom\StageB\*.safetensors")

        for i in customStageC:
            prior.append(i.split('\\')[-1])

        for i in customStageB:
            decoder.append(i.split('\\')[-1])
        return prior, decoder

    models_list_P, models_list_D = buildModelsLists ()

    def refreshModels ():
        prior, decoder = buildModelsLists ()
        return gr.Dropdown.update(choices=prior), gr.Dropdown.update(choices=decoder)

    def getGalleryIndex (evt: gr.SelectData):
        CascadeMemory.galleryIndex = evt.index

    def reuseLastSeed ():
        return CascadeMemory.lastSeed + CascadeMemory.galleryIndex
        
    def randomSeed ():
        return -1

    def i2iSetDimensions (image, w, h):
        #must be x128 to be safe, image will be resized to set width/height later
        if image is not None:
            w = 128 * (image.size[0] // 128)
            h = 128 * (image.size[1] // 128)
        return [w, h]

    def i2iImageFromGallery (gallery):
        try:
            newImage = gallery[CascadeMemory.galleryIndex][0]['name'].split('?')
            return newImage[0]
        except:
            return None

    def toggleKarras ():
        if CascadeMemory.karras == False:
            CascadeMemory.karras = True
            return gr.Button.update(value='\U0001D40A', variant='primary')
        else:
            CascadeMemory.karras = False
            return gr.Button.update(value='\U0001D542', variant='secondary')

    def toggleGenerate ():
        return gr.Button.update(value='...', variant='secondary', interactive=False)

    with gr.Blocks() as stable_cascade_block:
        with ResizeHandleRow():
            with gr.Column():
                with gr.Row():
                    modelP = gr.Dropdown(models_list_P, label='Model (Prior)', value="lite", type='value', scale=1)
                    refresh = ToolButton(value='\U0001f504')
                    modelD = gr.Dropdown(models_list_D, label='Model (Decoder)', value="lite", type='value', scale=1)
                    schedulerP = gr.Dropdown(["default",
                                             "DPM++ 2M",
                                             "DPM++ 2M SDE",
                                             "SA-solver",
                                             ],
                        label='Sampler (Prior)', value="default", type='value', scale=1)
                    karras = ToolButton(value="\U0001D542", variant='secondary', tooltip="use Karras sigmas")

                prompt = gr.Textbox(label='Prompt', placeholder='Enter a prompt here...', default='', lines=2)

                with gr.Row():
                    negative_prompt = gr.Textbox(label='Negative', placeholder='', lines=2)
                    style = gr.Dropdown([x[0] for x in styles.styles_list], label='Style', value="(None)", type='index', scale=0)
                with gr.Row():
                    width = gr.Slider(label='Width', minimum=128, maximum=4096, step=128, value=1024, elem_id="StableCascade_width")
                    swapper = ToolButton(value="\U000021C5")
                    height = gr.Slider(label='Height', minimum=128, maximum=4096, step=128, value=1024, elem_id="StableCascade_height")
                with gr.Row():
                    guidance_scale = gr.Slider(label='CFG', minimum=1, maximum=16, step=0.5, value=4.0)
                    prior_step = gr.Slider(label='Steps (Prior)', minimum=1, maximum=60, step=1, value=20)
                    decoder_steps = gr.Slider(label='Steps (Decoder)', minimum=1, maximum=20, step=1, value=10)
                with gr.Row():
                    sampling_seed = gr.Number(label='Seed', value=-1, precision=0, scale=2)
                    random = ToolButton(value="\U0001f3b2\ufe0f")
                    reuseSeed = ToolButton(value="\u267b\ufe0f")
                    batch_size = gr.Number(label='Batch Size', minimum=1, maximum=9, value=1, precision=0, scale=0)
#                with gr.Row():
#                    resolution = gr.Slider(label='Resolution multiple (prior)', minimum=32, maximum=64, step=0.01, value=42.67)
#                    latentScale = gr.Slider(label='Latent scale (VAE)', minimum=6, maximum=16, step=0.01, value=10.67)


                with gr.Accordion(label='Image prompt', open=False):
                    with gr.Row():
                        i2iSource = gr.Image(label='image source', sources=['upload'], type='pil', interactive=True, show_download_button=False)
                        with gr.Column():
                            i2iStrength = gr.Slider(label='Image Strength', minimum=0.00, maximum=2.0, step=0.01, value=1.0)
                            i2iSetWH = gr.Button(value='Set safe Width / Height from image')
                            i2iFromGallery = gr.Button(value='Get image from gallery')


                ctrls = [modelP, modelD, prompt, negative_prompt, width, height, guidance_scale, prior_step, decoder_steps,
                         sampling_seed, batch_size, schedulerP, style, i2iSource, i2iStrength]#, resolution, latentScale]

            with gr.Column():
                generate_button = gr.Button(value="Generate", variant='primary')
                output_gallery = gr.Gallery(label='Output', height=shared.opts.gallery_height or None, show_label=False,
                                            object_fit='contain', visible=True, columns=3, preview=True)
                
                with gr.Row():
                    buttons = parameters_copypaste.create_buttons(["img2img", "inpaint", "extras"])

                for tabname, button in buttons.items():
                    parameters_copypaste.register_paste_params_button(parameters_copypaste.ParamBinding(
                        paste_button=button, tabname=tabname, source_text_component=prompt, source_image_component=output_gallery,
                    ))

        refresh.click(refreshModels, inputs=[], outputs=[modelP, modelD])
        karras.click(toggleKarras, inputs=[], outputs=karras)
        swapper.click(fn=None, _js="function(){switchWidthHeight('StableCascade')}", inputs=None, outputs=None, show_progress=False)
        random.click(randomSeed, inputs=[], outputs=sampling_seed, show_progress=False)
        reuseSeed.click(reuseLastSeed, inputs=[], outputs=sampling_seed, show_progress=False)

        i2iSetWH.click (fn=i2iSetDimensions, inputs=[i2iSource, width, height], outputs=[width, height], show_progress=False)
        i2iFromGallery.click (fn=i2iImageFromGallery, inputs=[output_gallery], outputs=[i2iSource])

        output_gallery.select (fn=getGalleryIndex, inputs=[], outputs=[])

        generate_button.click(toggleGenerate, inputs=[], outputs=[generate_button])
        generate_button.click(predict, inputs=ctrls, outputs=[output_gallery, generate_button])
    return [(stable_cascade_block, "StableCascade", "stable_cascade")]

script_callbacks.on_ui_tabs(on_ui_tabs)
