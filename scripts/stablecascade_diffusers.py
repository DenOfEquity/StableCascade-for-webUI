from diffusers.utils import check_min_version
check_min_version("0.28.1")


class CascadeMemory:
    ModuleReload = False
    noUnload = False
    teCLIP = None
    lastPrior = None
    lastDecoder = None
    lastTextEncoder = None
    prior = None
    decoder = None
    lastSeed = -1
    galleryIndex = 0
    torchMessage = True     #   display information message about torch/bfloat16, set to False after first check
    locked = False  #   for preventing changes to the following volatile state while generating
    karras = False
    force_f16 = False
    embedsState = 0

import gc
import gradio
import numpy
from PIL import Image
import torch
try:
    from importlib import reload
    CascadeMemory.ModuleReload = True
except:
    CascadeMemory.ModuleReload = False

from modules import script_callbacks, images, shared
from modules.processing import get_fixed_seed
from modules.shared import opts
from modules.ui_components import ResizeHandleRow
import modules.infotext_utils as parameters_copypaste

from transformers import T5TokenizerFast, T5ForConditionalGeneration
from transformers import CLIPTextModelWithProjection, CLIPTokenizer
from diffusers import StableCascadeUNet, DDPMWuerstchenScheduler
from diffusers import DPMSolverSinglestepScheduler, DPMSolverMultistepScheduler, LCMScheduler, SASolverScheduler
from diffusers.pipelines.wuerstchen.modeling_paella_vq_model import PaellaVQModel
from diffusers import AutoencoderKL
from diffusers.utils import logging

import customStylesListSC as styles
import modelsListSC as models
import scripts.SC_pipeline as pipeline

# modules/processing.py
def create_infotext(priorModel, decoderModel, vaeModel, positive_prompt, negative_prompt, clipskip, guidance_scale, prior_steps, decoder_steps, seed, schedulerP, schedulerD, width, height, ):
    karras = " : Karras" if CascadeMemory.karras == True else ""
    generation_params = {
        "Size"                      : f"{width}x{height}",
        "Seed"                      : seed,
        "Scheduler(Prior/Decoder)"  : f"{schedulerP}/{schedulerD}{karras}",
        "Steps(Prior/Decoder)"      : f"{prior_steps}/{decoder_steps}",
        "CFG"                       : guidance_scale,
        "CLIP skip"                 : clipskip,
    }

    model_text = "(" + priorModel.split('.')[0] + "/" + decoderModel.split('.')[0] + "/" + vaeModel + ")"
    
    prompt_text = f"Prompt: {positive_prompt}"
    if negative_prompt != "":
        prompt_text += (f"\nNegative: {negative_prompt}")
    generation_params_text = ", ".join([k if k == v else f'{k}: {v}' for k, v in generation_params.items() if v is not None])

    return f"Model: StableCascade {model_text}\n{prompt_text}\n{generation_params_text}"


def predict(priorModel, decoderModel, vaeModel, positive_prompt, negative_prompt, clipskip,  width, height, guidance_scale,
            prior_steps, decoder_steps, seed, num_images, PriorScheduler, DecoderScheduler, style, i2iSource1, i2iSource2):
            #resolution, latentScale):
    logging.set_verbosity(logging.ERROR)

    torch.set_grad_enabled(False)

    if style != 0:
        positive_prompt = styles.styles_list[style][1].replace("{prompt}", positive_prompt)
        negative_prompt = negative_prompt + styles.styles_list[style][2]

    gc.collect()
    torch.cuda.empty_cache()

    fixed_seed = get_fixed_seed(seed)
    CascadeMemory.lastSeed = fixed_seed

    useLitePrior = "lite" in priorModel
    useLiteDecoder = "lite" in decoderModel

    if CascadeMemory.force_f16 == True:
        dtype = torch.float16
    elif torch.cuda.is_bf16_supported() == True and int(torch.__version__[0]) >= 2 and int(torch.__version__[2]) >= 2:
        dtype = torch.bfloat16
    else:
        if CascadeMemory.torchMessage == True:
            if torch.cuda.is_bf16_supported() == True:
                print ("INFO: StableCascade: Using float16. Hardware supports bfloat16, but needs Torch version >= 2.2.0 (using " + torch.__version__ + ").")
            else:
                print ("INFO: StableCascade: Using float16. Hardware does not support bfloat16.")
            CascadeMemory.torchMessage = False
        dtype = torch.float16

    ####   image embeds, basically using images to prompt - not image to image
    image_embeds0 = torch.zeros(
        num_images,
        1,
        768,
        device='cpu',
        dtype=torch.float32,
    )
    image_embeds0 = image_embeds0.to('cuda').to(dtype)
    if i2iSource1 or i2iSource2:
        prior = pipeline.StableCascadePriorPipeline_DoE.from_pretrained(
            "stabilityai/stable-cascade-prior", 
            local_files_only=False, cache_dir=".//models//diffusers//",
            prior=None,
            text_encoder=None,
            tokenizer=None,
            scheduler=None,
            variant="bf16",
            torch_dtype=torch.float32)

        if i2iSource1:
            image_embeds1, _ = prior.encode_image(images=[i2iSource1], device='cpu', dtype=torch.float32, batch_size=1, num_images_per_prompt=1)
            image_embeds1 = image_embeds1.to('cuda').to(dtype)
            del i2iSource1
        else:
            image_embeds1 = image_embeds0

        if i2iSource2:
            image_embeds2, _ = prior.encode_image(images=[i2iSource2], device='cpu', dtype=torch.float32, batch_size=1, num_images_per_prompt=1)
            image_embeds2 = image_embeds2.to('cuda').to(dtype)
        else:
            image_embeds2 = image_embeds0

        del prior
        
        match CascadeMemory.embedsState:
            case 3:         #   0b11: both negative
                positive_image_embeds = torch.cat((image_embeds0, image_embeds0), dim=1)
                negative_image_embeds = torch.cat((image_embeds1, image_embeds2), dim=1)
            case 2:         #   0b10: 1 negative, 2 positive
                positive_image_embeds = image_embeds2
                negative_image_embeds = image_embeds1
            case 1:         #   0b01, 1 positive, 2 negative
                positive_image_embeds = image_embeds1
                negative_image_embeds = image_embeds2
            case 0:         #   0b00,   both positive
                positive_image_embeds = torch.cat((image_embeds1, image_embeds2), dim=1)
                negative_image_embeds = torch.cat((image_embeds0, image_embeds0), dim=1)

        del image_embeds1, image_embeds2
    else:
        positive_image_embeds = image_embeds0
        negative_image_embeds = image_embeds0
    del image_embeds0
        
    ####    note: image_embeds are repeated for num_images in pipeline
    ####    end image embeds

    ####   text encoder
    source = priorModel if (priorModel in models.models_list_prior) else "stabilityai/stable-cascade-prior"
    tokenizer = CLIPTokenizer.from_pretrained(
        source,
        subfolder='tokenizer',
        local_files_only=False, cache_dir=".//models//diffusers//",
        torch_dtype=dtype)

    # def prompt_and_weights (tokenizer, prompt):
        # promptSplit = prompt.split('|')
        # newPrompt = []
        # weights = []
        # max_length = tokenizer.model_max_length
        
        # for s in promptSplit:
            # subpromptSplit = s.strip().split(' ')
            # cleanedPrompt = ' '.join((t.split(':')[0] for t in subpromptSplit))
            # newPrompt.append(cleanedPrompt)

            # subWeights = [1.0]
 
            # for t in subpromptSplit:
                # t = t.split(':')
                # if len(t) == 1:
                    # weight = 1.0
                # elif t[1] == '':
                    # weight = 1.0
                # else:
                    # try:
                        # weight = float(t[1].rstrip(','))
                    # except:
                        # weight = 1.0
     
                # text_inputs = tokenizer(
                    # t[0],
                    # padding=False,
                    # max_length=max_length,
                    # truncation=True,
                    # return_attention_mask=False,
                    # add_special_tokens=False,
                    # return_tensors="pt",
                # )
     
                # tokenLength = len(text_inputs.input_ids[0])
                # for w in range(tokenLength):
                    # subWeights.append(weight)
                    
            # weights.append(subWeights)
        # return newPrompt, weights

    # fixed_positive_prompt, positive_weights = prompt_and_weights(tokenizer, positive_prompt)
    # fixed_negative_prompt, negative_weights = prompt_and_weights(tokenizer, negative_prompt)

    # while len(fixed_positive_prompt) < len(fixed_negative_prompt):
        # fixed_positive_prompt.append('')
        # positive_weights.append([1.0])
    # while len(fixed_positive_prompt) > len(fixed_negative_prompt):
        # fixed_negative_prompt.append('')
        # negative_weights.append([1.0])

    # text_inputs = tokenizer(
        # fixed_positive_prompt + fixed_negative_prompt,
        # padding=True,
        # max_length=tokenizer.model_max_length,
        # truncation=True,
        # return_attention_mask=True,
        # return_tensors="pt",
    # )
        
    # positive_input_ids = text_inputs.input_ids[0:len(fixed_positive_prompt)]
    # negative_input_ids = text_inputs.input_ids[len(fixed_positive_prompt):]

    # positive_attention = text_inputs.attention_mask[0:len(fixed_positive_prompt)]
    # negative_attention = text_inputs.attention_mask[len(fixed_positive_prompt):]


    text_inputs = tokenizer(
        [positive_prompt] + [negative_prompt],
        padding=True,
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    positive_input_ids = text_inputs.input_ids[0:1]
    negative_input_ids = text_inputs.input_ids[1:]
    positive_attention = text_inputs.attention_mask[0:1]
    negative_attention = text_inputs.attention_mask[1:]

    del text_inputs
    del tokenizer

    if CascadeMemory.teCLIP == None or source != CascadeMemory.lastTextEncoder:
        try:
            CascadeMemory.teCLIP = CLIPTextModelWithProjection.from_pretrained(
                source, 
                subfolder='text_encoder',
                local_files_only=False, cache_dir=".//models//diffusers//",
                variant='bf16',
                torch_dtype=dtype)
        except:
            try:
                CascadeMemory.teCLIP = CLIPTextModelWithProjection.from_pretrained(
                    source, 
                    subfolder='text_encoder',
                    local_files_only=False, cache_dir=".//models//diffusers//",
                    torch_dtype=dtype)
            except:
                CascadeMemory.teCLIP = CLIPTextModelWithProjection.from_pretrained(
                    "stabilityai/stable-cascade-prior", 
                    subfolder='text_encoder',
                    local_files_only=False, cache_dir=".//models//diffusers//",
                    variant='bf16',
                    torch_dtype=dtype)
        CascadeMemory.lastTextEncoder = source

    CascadeMemory.teCLIP.cuda()

    text_encoder_output = CascadeMemory.teCLIP(
        positive_input_ids.to('cuda'), attention_mask=positive_attention.to('cuda'), output_hidden_states=True
    )
    positive_embeds = text_encoder_output.hidden_states[-(clipskip+1)]
    positive_pooled = text_encoder_output.text_embeds.unsqueeze(1)

    # positive_mean_before = positive_embeds.mean()
    # for l in range(len(positive_embeds)):
        # for p in range(min(77, len(positive_weights[l]))):
            # positive_embeds[l][p] *= positive_weights[l][p]
    # positive_mean_after = positive_embeds.mean()
    # positive_embeds *= positive_mean_before / positive_mean_after

    positive_embeds = positive_embeds.view(1, -1, 1280)
    positive_pooled = positive_pooled[0].unsqueeze(0)

    positive_embeds = positive_embeds.to(dtype=dtype, device='cuda')
    positive_pooled = positive_pooled.to(dtype=dtype, device='cuda')
    positive_embeds = positive_embeds.repeat_interleave(num_images, dim=0)
    positive_pooled = positive_pooled.repeat_interleave(num_images, dim=0)

    if guidance_scale > 1.0:
        text_encoder_output = CascadeMemory.teCLIP(
            negative_input_ids.to('cuda'), attention_mask=negative_attention.to('cuda'), output_hidden_states=True
        )
        negative_embeds = text_encoder_output.hidden_states[-1]
        negative_pooled = text_encoder_output.text_embeds.unsqueeze(1)

        # negative_mean_before = negative_embeds.mean()
        # for l in range(len(negative_embeds)):
            # for p in range(min(77, len(negative_weights[l]))):
                # negative_embeds[l][p] *= negative_weights[l][p]
        # negative_mean_after = negative_embeds.mean()
        # negative_embeds *= negative_mean_before / negative_mean_after

        negative_embeds = negative_embeds.view(1, -1, 1280)
        negative_pooled = negative_pooled[0].unsqueeze(0)
        negative_embeds = negative_embeds.to(dtype=dtype, device='cuda')
        negative_pooled = negative_pooled.to(dtype=dtype, device='cuda')
        negative_embeds = negative_embeds.repeat_interleave(num_images, dim=0)
        negative_pooled = negative_pooled.repeat_interleave(num_images, dim=0)
    else:
        negative_embeds = None
        negative_pooled = None

    del positive_input_ids, negative_input_ids, positive_attention, negative_attention

    if CascadeMemory.noUnload:
        pass#CascadeMemory.teCLIP.cpu() #   try keeping on GPU to free memory to store full unet
    else:
        CascadeMemory.teCLIP = None
    ####    end text_encoder

    ####    setup prior pipeline
    if CascadeMemory.prior == None:
        CascadeMemory.prior = pipeline.StableCascadePriorPipeline_DoE.from_pretrained(
            "stabilityai/stable-cascade-prior", 
            local_files_only=False, cache_dir=".//models//diffusers//",
            image_encoder=None, feature_extractor=None, tokenizer=None, text_encoder=None,
            prior=None,
            variant='bf16',
            torch_dtype=dtype,)
    ####    end setup prior pipeline

    ####    get prior unet
    if not CascadeMemory.noUnload or priorModel != CascadeMemory.lastPrior:
        print ("StableCascade: loading prior unet ...", end="\r", flush=True)
        if priorModel in models.models_list_prior:
        #   custom diffusers type
            CascadeMemory.prior.prior = StableCascadeUNet.from_pretrained(
                priorModel, 
                subfolder="prior_lite" if "lite" in priorModel else "prior",
                local_files_only=False, cache_dir=".//models//diffusers//",
                use_low_cpu_mem=True,
                torch_dtype=dtype)
        elif priorModel == "lite":
            CascadeMemory.prior.prior = StableCascadeUNet.from_pretrained("stabilityai/stable-cascade-prior", 
                local_files_only=False, cache_dir=".//models//diffusers//",
                subfolder="prior_lite",
                variant="bf16",
                use_low_cpu_mem=True,
                torch_dtype=dtype)
        elif priorModel == "full":
            CascadeMemory.prior.prior = StableCascadeUNet.from_pretrained("stabilityai/stable-cascade-prior", 
                local_files_only=False, cache_dir=".//models//diffusers//",
                subfolder="prior",
                variant="bf16",
                use_low_cpu_mem=True,
                torch_dtype=dtype)
        else:# ".safetensors" in priorModel:
            customStageC = ".//models//diffusers//StableCascadeCustom//StageC//" + priorModel
            CascadeMemory.prior.prior = StableCascadeUNet.from_single_file(
                customStageC,
                local_files_only=True, cache_dir=".//models//diffusers//",
                use_safetensors=True,
                subfolder="prior_lite" if "lite" in priorModel else "prior",
                use_low_cpu_mem=True,
                torch_dtype=dtype,
                config="stabilityai/stable-cascade-prior")

        CascadeMemory.prior.prior.to(memory_format=torch.channels_last)
        CascadeMemory.lastPrior = priorModel if CascadeMemory.noUnload else None
    ####    end get prior unet

    if useLitePrior == False:
        CascadeMemory.prior.enable_sequential_cpu_offload()       # good for full models on 8GB, but unnecessary for lite (and slows down generation)
    else:
        CascadeMemory.prior.to('cuda')

    generator = [torch.Generator(device="cpu").manual_seed(fixed_seed+i) for i in range(num_images)]

    schedulerConfig = dict(CascadeMemory.prior.scheduler.config)
    schedulerConfig['use_karras_sigmas'] = CascadeMemory.karras
    schedulerConfig['clip_sample'] = False
    schedulerConfig.pop('algorithm_type', None) 

    if PriorScheduler == 'DPM++ 2M':
        CascadeMemory.prior.scheduler = DPMSolverMultistepScheduler.from_config(schedulerConfig)
    elif PriorScheduler == "DPM++ 2M SDE":
        schedulerConfig['algorithm_type'] = 'sde-dpmsolver++'
        CascadeMemory.prior.scheduler = DPMSolverMultistepScheduler.from_config(schedulerConfig)
    elif PriorScheduler == "LCM":
        CascadeMemory.prior.scheduler = LCMScheduler.from_config(schedulerConfig)
    elif PriorScheduler == "SA-solver":
        schedulerConfig['algorithm_type'] = 'data_prediction'
        CascadeMemory.prior.scheduler = SASolverScheduler.from_config(schedulerConfig)
    else:
        CascadeMemory.prior.scheduler = DDPMWuerstchenScheduler.from_config(schedulerConfig)

    with torch.inference_mode():
        prior_output = CascadeMemory.prior(
            prompt_embeds                   = positive_embeds,
            prompt_embeds_pooled            = positive_pooled,
            negative_prompt_embeds          = negative_embeds,
            negative_prompt_embeds_pooled   = negative_pooled,

            image_embeds=positive_image_embeds,
            negative_image_embeds=negative_image_embeds,

            width=width,
            height=height,
            guidance_scale=guidance_scale,
            num_inference_steps=prior_steps,
            num_images_per_prompt=num_images,
            generator=generator,
        )

    del generator
    
    if not CascadeMemory.noUnload:
        CascadeMemory.prior.prior= None
        CascadeMemory.lastPrior = None

    positive_embeds = prior_output.get("prompt_embeds", None)
    positive_pooled = prior_output.get("prompt_embeds_pooled", None)
    negative_embeds = prior_output.get("negative_prompt_embeds", None)
    negative_pooled = prior_output.get("negative_prompt_embeds_pooled", None)
#i: (num output images, num input images, 768)
#e: (num output images, 77, 1280)
#p: (num output images, 1, 1280)

    gc.collect()
    torch.cuda.empty_cache()

    ####    setup decoder pipeline
    if CascadeMemory.decoder == None:
        CascadeMemory.decoder = pipeline.StableCascadeDecoderPipeline_DoE.from_pretrained(
            "stabilityai/stable-cascade", 
            local_files_only=False, cache_dir=".//models//diffusers//",
            decoder=None, 
            vqgan=None,
            variant='bf16',
            torch_dtype=dtype,)
    ####    end setup decoder pipeline

    ####    get decoder unet
    if not CascadeMemory.noUnload or decoderModel != CascadeMemory.lastDecoder:
        print ("StableCascade: loading decoder unet ...", end="\r", flush=True)
        if decoderModel in models.models_list_decoder:
        #   custom diffusers type
            CascadeMemory.decoder.decoder = StableCascadeUNet.from_pretrained(
                decoderModel, 
                subfolder="decoder_lite" if "lite" in decoderModel else "decoder",
                local_files_only=False, cache_dir=".//models//diffusers//",
                use_low_cpu_mem=True,
                torch_dtype=dtype)
        elif decoderModel == "lite":
            CascadeMemory.decoder.decoder = StableCascadeUNet.from_pretrained(
                "stabilityai/stable-cascade", 
                local_files_only=False, cache_dir=".//models//diffusers//",
                subfolder="decoder_lite",
                variant="bf16",
                use_low_cpu_mem=True,
                torch_dtype=dtype)
        elif decoderModel == "full":
            CascadeMemory.decoder.decoder = StableCascadeUNet.from_pretrained(
                "stabilityai/stable-cascade", 
                local_files_only=False, cache_dir=".//models//diffusers//",
                subfolder="decoder",
                variant="bf16",
                use_low_cpu_mem=True,
                torch_dtype=dtype)
        else:# ".safetensors" in decoderModel:
            customStageC = ".//models//diffusers//StableCascadeCustom//StageC//" + decoderModel
            CascadeMemory.decoder.decoder = StableCascadeUNet.from_single_file(
                customStageC,
                local_files_only=True, cache_dir=".//models//diffusers//",
                use_safetensors=True,
                subfolder="decoder_lite" if "lite" in decoderModel else "decoder",
                use_low_cpu_mem=True,
                torch_dtype=dtype,
                config="stabilityai/stable-cascade")

        CascadeMemory.decoder.decoder.to(memory_format=torch.channels_last)
        CascadeMemory.lastDecoder = decoderModel if CascadeMemory.noUnload else None
    ####    end get decoder unet

    ####    VAE always loaded - it's only 35MB
    if vaeModel == 'madebyollin':
        # Load the Stage-A-ft-HQ model
        CascadeMemory.decoder.vqgan = PaellaVQModel.from_pretrained("madebyollin/stage-a-ft-hq", 
                                              local_files_only=False, cache_dir=".//models//diffusers//", torch_dtype=dtype)
    else:
        #default
        CascadeMemory.decoder.vqgan = PaellaVQModel.from_pretrained("stabilityai/stable-cascade", 
                                              local_files_only=False, cache_dir=".//models//diffusers//", subfolder="vqgan", torch_dtype=dtype)

    CascadeMemory.decoder.enable_model_cpu_offload()

    ##  regenerate the Generator, needed for deterministic outputs - reusing from earlier doesn't work
        #still not correct with custom checkpoint?
    generator = [torch.Generator(device="cpu").manual_seed(fixed_seed+i) for i in range(num_images)]

    #   trying to colour the noise here is 100% ineffective

    schedulerConfig = dict(CascadeMemory.decoder.scheduler.config)
    schedulerConfig['use_karras_sigmas'] = CascadeMemory.karras
    schedulerConfig['clip_sample'] = False
    schedulerConfig.pop('algorithm_type', None) 

    if DecoderScheduler == 'DPM++ 2M':
        CascadeMemory.decoder.scheduler = DPMSolverMultistepScheduler.from_config(schedulerConfig)
    elif DecoderScheduler == "DPM++ 2M SDE":
        schedulerConfig['algorithm_type'] = 'sde-dpmsolver++'
        CascadeMemory.decoder.scheduler = DPMSolverMultistepScheduler.from_config(schedulerConfig)
    elif DecoderScheduler == "LCM":
        CascadeMemory.decoder.scheduler = LCMScheduler.from_config(schedulerConfig)
    elif DecoderScheduler == "SA-solver":
        schedulerConfig['algorithm_type'] = 'data_prediction'
        CascadeMemory.decoder.scheduler = SASolverScheduler.from_config(schedulerConfig)
    else:
        CascadeMemory.decoder.scheduler = DDPMWuerstchenScheduler.from_config(schedulerConfig)

    with torch.inference_mode():
        decoder_output = CascadeMemory.decoder(
            image_embeddings=prior_output.image_embeddings.to(dtype),
            prompt_embeds                   = positive_embeds,
            prompt_embeds_pooled            = positive_pooled,
            negative_prompt_embeds          = negative_embeds,
            negative_prompt_embeds_pooled   = negative_pooled,
            prompt=None,
            negative_prompt=None,
            guidance_scale=1,
            output_type="pil",
            num_inference_steps=decoder_steps,
            generator=generator,
        ).images

    del prior_output, positive_embeds, positive_pooled, negative_embeds, negative_pooled
    del generator
    
    if not CascadeMemory.noUnload:
        CascadeMemory.decoder.decoder = None
        CascadeMemory.decoder.vqgan = None
        CascadeMemory.lastDecoder = None

    gc.collect()
    torch.cuda.empty_cache()

    result = []

    for image in decoder_output:
        info=create_infotext(priorModel, decoderModel, vaeModel, positive_prompt, negative_prompt, clipskip, guidance_scale, prior_steps, decoder_steps, fixed_seed,
                             PriorScheduler, DecoderScheduler, width, height)
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

    CascadeMemory.locked = False
    return gradio.Button.update(value='Generate', variant='primary', interactive=True), gradio.Button.update(interactive=True), result


def on_ui_tabs():
    if CascadeMemory.ModuleReload:
        reload (pipeline)
        reload (models)
        reload (styles)

    from modules.ui_components import ToolButton                                                     

    def buildModelsLists ():
        prior = ["lite", "full"] + models.models_list_prior
        decoder = ["lite", "full"] + models.models_list_decoder
        
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
        return gradio.Dropdown.update(choices=prior), gradio.Dropdown.update(choices=decoder)

    def getGalleryIndex (evt: gradio.SelectData):
        CascadeMemory.galleryIndex = evt.index

    def reuseLastSeed ():
        return CascadeMemory.lastSeed + CascadeMemory.galleryIndex
        
    def randomSeed ():
        return -1

    def i2iImageFromGallery (gallery):
        try:
            newImage = gallery[CascadeMemory.galleryIndex][0]['name'].split('?')
            return newImage[0]
        except:
            return None

    def i2iSwap (i1, i2):
        return i2, i1

    def toggleNU ():
        if not CascadeMemory.locked:
            CascadeMemory.noUnload ^= True
        return gradio.Button.update(variant=['secondary', 'primary'][CascadeMemory.noUnload])
    def unloadM ():
        if not CascadeMemory.locked:
            CascadeMemory.teCLIP = None
            CascadeMemory.prior = None
            CascadeMemory.decoder = None
            CascadeMemory.lastPrior = None
            CascadeMemory.lastDecoder = None
            CascadeMemory.lastTextEncoder = None
            gc.collect()
            torch.cuda.empty_cache()
        else:
            gradio.Info('Unable to unload models while using them.')
    def clearE ():
        if CascadeMemory.locked:
            CascadeMemory.locked = False
            return gradio.Button.update(value='Generate', variant='primary', interactive=True)
        
    def toggleSP ():
        if not CascadeMemory.locked:
            return gradio.Button.update(variant='primary')
    def superPrompt (prompt, seed):
        tokenizer = getattr (shared, 'SuperPrompt_tokenizer', None)
        superprompt = getattr (shared, 'SuperPrompt_model', None)
        if tokenizer is None:
            tokenizer = T5TokenizerFast.from_pretrained(
                'roborovski/superprompt-v1',
                cache_dir='.//models//diffusers//',
            )
            shared.SuperPrompt_tokenizer = tokenizer
        if superprompt is None:
            superprompt = T5ForConditionalGeneration.from_pretrained(
                'roborovski/superprompt-v1',
                cache_dir='.//models//diffusers//',
                device_map='auto',
                torch_dtype=torch.float16
            )
            shared.SuperPrompt_model = superprompt
            print("SuperPrompt-v1 model loaded successfully.")
            if torch.cuda.is_available():
                superprompt.to('cuda')

        torch.manual_seed(get_fixed_seed(seed))
        device = superprompt.device
        systemprompt1 = "Expand the following prompt to add more detail: "
        
        input_ids = tokenizer(systemprompt1 + prompt, return_tensors="pt").input_ids.to(device)
        outputs = superprompt.generate(input_ids, max_new_tokens=77, repetition_penalty=1.2, do_sample=True)
        dirty_text = tokenizer.decode(outputs[0])
        result = dirty_text.replace("<pad>", "").replace("</s>", "").strip()
        
        return gradio.Button.update(variant='secondary'), result


    def toggleKarras ():
        if not CascadeMemory.locked:
            CascadeMemory.karras ^= True
        return gradio.Button.update(variant='primary' if CascadeMemory.karras == True else 'secondary',
                                value='\U0001D40A' if CascadeMemory.karras == True else '\U0001D542')
    def toggleF16 ():
        if not CascadeMemory.locked:
            CascadeMemory.force_f16 ^= True
        return gradio.Button.update(variant='primary' if CascadeMemory.force_f16 == True else 'secondary')


    def toggleE1 ():
        if not CascadeMemory.locked:
            CascadeMemory.embedsState ^= 2
        return gradio.Button.update(variant='primary' if (CascadeMemory.embedsState & 2) else 'secondary')
    def toggleE2 ():
        if not CascadeMemory.locked:
            CascadeMemory.embedsState ^= 1
        return gradio.Button.update(variant='primary' if (CascadeMemory.embedsState & 1) else 'secondary')
            
    def toggleGenerate ():
        CascadeMemory.locked = True
        return gradio.Button.update(value='...', variant='secondary', interactive=False), gradio.Button.update(interactive=False)

    schedulerList = ["default", "DPM++ 2M", "DPM++ 2M SDE", "LCM", "SA-solver", ]

    def parsePrompt (positive, negative, clipskip, width, height, seed, schedulerP, schedulerD, stepsP, stepsD, cfg):
        p = positive.split('\n')
        lineCount = len(p)

        negative = ''
        
        if "Prompt" != p[0] and "Prompt: " != p[0][0:8]:               #   civitAI style special case
            positive = p[0]
            l = 1
            while (l < lineCount) and not (p[l][0:17] == "Negative prompt: " or p[l][0:7] == "Steps: " or p[l][0:6] == "Size: "):
                if p[l] != '':
                    positive += '\n' + p[l]
                l += 1
        
        for l in range(lineCount):
            if "Prompt" == p[l][0:6]:
                if ": " == p[l][6:8]:                                   #   mine
                    positive = str(p[l][8:])
                    c = 1
                elif "Prompt" == p[l] and (l+1 < lineCount):            #   webUI
                    positive = p[l+1]
                    c = 2
                else:
                    continue

                while (l+c < lineCount) and not (p[l+c][0:10] == "Negative: " or p[l+c][0:15] == "Negative Prompt" or p[l+c] == "Params" or p[l+c][0:7] == "Steps: " or p[l+c][0:6] == "Size "):
                    if p[l+c] != '':
                        positive += '\n' + p[l+c]
                    c += 1
                l += 1

            elif "Negative" == p[l][0:8]:
                if ": " == p[l][8:10]:                                  #   mine
                    negative = str(p[l][10:])
                    c = 1
                elif " prompt: " == p[l][8:17]:                         #   civitAI
                    negative = str(p[l][17:])
                    c = 1
                elif " Prompt" == p[l][8:15] and (l+1 < lineCount):     #   webUI
                    negative = p[l+1]
                    c = 2
                else:
                    continue
                
                while (l+c < lineCount) and not (p[l+c] == "Params" or p[l+c][0:7] == "Steps: " or p[l+c][0:6] == "Size: "):
                    if p[l+c] != '':
                        negative += '\n' + p[l+c]
                    c += 1
                l += 1

            else:
                params = p[l].split(',')
                for k in range(len(params)):
                    pairs = params[k].strip().split(' ')
                    match pairs[0]:
                        case "Size:":
                            size = pairs[1].split('x')
                            width = 128 * ((int(size[0]) + 64) // 128)
                            height = 128 * ((int(size[1]) + 64) // 128)
                        case "Seed:":
                            seed = int(pairs[1])
                        case "Sampler:":
                            sched = ' '.join(pairs[1:])
                            if sched in schedulerList:
                                scheduler = sched
                        case "Scheduler(Prior/Decoder):":
                            sched = ' '.join(pairs[1:])
                            sched = sched.split('/')
                            if sched[0] in schedulerList:
                                schedulerP = sched[0]
                            if sched[1] in schedulerList:
                                schedulerD = sched[1]
                        case "Scheduler:":
                            sched = ' '.join(pairs[1:])
                            if sched in schedulerList:
                                schedulerP = sched
                        case "Steps(Prior/Decoder):":
                            steps = str(pairs[1]).split('/')
                            stepsP = int(steps[0])
                            stepsD = int(steps[1])
                        case "Steps:":
                            stepsP = int(pairs[1])
                        case "CFG":
                            if "scale:" == pairs[1]:
                                cfg = float(pairs[2])
                        case "CFG:":
                            cfg = float(pairs[1])
                        case "width:":
                            width = 128 * ((int(pairs[1]) + 64) // 128)
                        case "height:":
                            height = 128 * ((int(pairs[1]) + 64) // 128)
                        case "CLIP skip:":
                            clipskip = int(pairs[1])
        return positive, negative, clipskip, width, height, seed, schedulerP, schedulerD, stepsP, stepsD, cfg



    with gradio.Blocks() as stable_cascade_block:
        with ResizeHandleRow():
            with gradio.Column():
                with gradio.Row():
                    refresh = ToolButton(value='\U0001f504')
                    modelP = gradio.Dropdown(models_list_P, label='Stage C (Prior)', value="lite", type='value', scale=2)
                    modelD = gradio.Dropdown(models_list_D, label='Stage B (Decoder)', value="lite", type='value', scale=2)
                    modelV = gradio.Dropdown(['default', 'madebyollin'], label='Stage A (VAE)', value='default', type='value', scale=0)
                    clipskip = gradio.Number(label='CLIP skip', minimum=0, maximum=2, step=1, value=0, precision=0, scale=1)

                with gradio.Row():
                    parse = ToolButton(value="↙️", variant='secondary', tooltip="parse")
                    SP = ToolButton(value='ꌗ', variant='secondary', tooltip='zero out negative embeds')
                    karras = ToolButton(value="\U0001D542", variant='secondary', tooltip="use Karras sigmas")
                    schedulerP = gradio.Dropdown(schedulerList, label='Sampler (Prior)', value="default", type='value', scale=1)
                    schedulerD = gradio.Dropdown(schedulerList, label='Sampler (Decoder)', value="default", type='value', scale=1)
                    style = gradio.Dropdown([x[0] for x in styles.styles_list], label='Style', value="(None)", type='index', scale=1)
                    f16 = ToolButton(value="f16", variant='secondary', tooltip="force float16")

                with gradio.Row():
                    prompt = gradio.Textbox(label='Prompt', placeholder='Enter a prompt here...', default='', lines=2)

                with gradio.Row():
                    negative_prompt = gradio.Textbox(label='Negative', placeholder='', lines=1.0)
                with gradio.Row():
                    width = gradio.Slider(label='Width', minimum=128, maximum=4096, step=128, value=1024, elem_id="StableCascade_width")
                    swapper = ToolButton(value="\U000021C4")
                    height = gradio.Slider(label='Height', minimum=128, maximum=4096, step=128, value=1024, elem_id="StableCascade_height")
                with gradio.Row():
                    prior_steps = gradio.Slider(label='Steps (Prior)', minimum=1, maximum=60, step=1, value=20)
                    decoder_steps = gradio.Slider(label='Steps (Decoder)', minimum=1, maximum=40, step=1, value=10)
                with gradio.Row():
                    guidance_scale = gradio.Slider(label='CFG', minimum=1, maximum=16, step=0.1, value=4.0)
                    sampling_seed = gradio.Number(label='Seed', value=-1, precision=0, scale=0)
                    random = ToolButton(value="\U0001f3b2\ufe0f")
                    reuseSeed = ToolButton(value="\u267b\ufe0f")
                    batch_size = gradio.Number(label='Batch Size', minimum=1, maximum=9, value=1, precision=0, scale=0)
#                with gradio.Row():
#                    resolution = gradio.Slider(label='Resolution multiple (prior)', minimum=32, maximum=64, step=0.01, value=42.67)
#                    latentScale = gradio.Slider(label='Latent scale (VAE)', minimum=6, maximum=16, step=0.01, value=10.67)

                with gradio.Accordion(label='Image prompt', open=False):
#add start/end? would need to modify pipeline

                    with gradio.Row():
                        i2iSource1 = gradio.Image(label='image source', sources=['upload'], type='pil', interactive=True, show_download_button=False)
                        i2iSource2 = gradio.Image(sources=['upload'], type='pil', interactive=True, show_download_button=False)
                    with gradio.Row():
                        embed1State = ToolButton('Neg', variant='secondary')
                        i2iFromGallery1 = gradio.Button(value='Get image (1) from gallery', scale=6)
                        i2iFromGallery2 = gradio.Button(value='Get image (2) from gallery', scale=6)
                        embed2State = ToolButton('Neg', variant='secondary')
                    with gradio.Row():
                        swapImages = gradio.Button(value='Swap images')

                with gradio.Row():
                    noUnload = gradio.Button(value='keep models loaded', variant='primary' if CascadeMemory.noUnload else 'secondary', tooltip='noUnload', scale=1)
                    unloadModels = gradio.Button(value='unload models', tooltip='force unload of models', scale=1)
#                    clearError = gradio.Button(value='remove Error', tooltip='clear Error', scale=1)

                ctrls = [modelP, modelD, modelV, prompt, negative_prompt, clipskip, width, height, guidance_scale, prior_steps, decoder_steps,
                         sampling_seed, batch_size, schedulerP, schedulerD, style, i2iSource1, i2iSource2]#, resolution, latentScale]

            with gradio.Column():
                generate_button = gradio.Button(value="Generate", variant='primary')
                output_gallery = gradio.Gallery(label='Output', height="75vh", show_label=False,
                                            object_fit='contain', visible=True, columns=1, preview=True)
                
                with gradio.Row():
                    buttons = parameters_copypaste.create_buttons(["img2img", "inpaint", "extras"])

                for tabname, button in buttons.items():
                    parameters_copypaste.register_paste_params_button(parameters_copypaste.ParamBinding(
                        paste_button=button, tabname=tabname, source_text_component=prompt, source_image_component=output_gallery,
                    ))

        noUnload.click(toggleNU, inputs=[], outputs=noUnload)
        unloadModels.click(unloadM, inputs=[], outputs=[], show_progress=True)
#        clearError.click(clearE, inputs=[], outputs=[generate_button])

        SP.click(toggleSP, inputs=[], outputs=SP)
        SP.click(superPrompt, inputs=[prompt, sampling_seed], outputs=[SP, prompt])

        parse.click(parsePrompt, inputs=[prompt, negative_prompt, clipskip, width, height, sampling_seed, schedulerP, schedulerD, prior_steps, decoder_steps, guidance_scale], outputs=[prompt, negative_prompt, clipskip, width, height, sampling_seed, schedulerP, schedulerD, prior_steps, decoder_steps, guidance_scale], show_progress=False)
        refresh.click(refreshModels, inputs=[], outputs=[modelP, modelD])
        karras.click(toggleKarras, inputs=[], outputs=karras)
        f16.click(toggleF16, inputs=[], outputs=f16)
        swapper.click(fn=None, _js="function(){switchWidthHeight('StableCascade')}", inputs=None, outputs=None, show_progress=False)
        random.click(randomSeed, inputs=[], outputs=sampling_seed, show_progress=False)
        reuseSeed.click(reuseLastSeed, inputs=[], outputs=sampling_seed, show_progress=False)

        i2iFromGallery1.click (fn=i2iImageFromGallery, inputs=[output_gallery], outputs=[i2iSource1])
        i2iFromGallery2.click (fn=i2iImageFromGallery, inputs=[output_gallery], outputs=[i2iSource2])
        swapImages.click (fn=i2iSwap, inputs=[i2iSource1, i2iSource2], outputs=[i2iSource1, i2iSource2])
        embed1State.click(fn=toggleE1, inputs=[], outputs=[embed1State], show_progress=False)
        embed2State.click(fn=toggleE2, inputs=[], outputs=[embed2State], show_progress=False)
        output_gallery.select (fn=getGalleryIndex, inputs=[], outputs=[])

        generate_button.click(predict, inputs=ctrls, outputs=[generate_button, SP, output_gallery])
        generate_button.click(toggleGenerate, inputs=[], outputs=[generate_button, SP])
    return [(stable_cascade_block, "StableCascade", "stable_cascade_DoE")]

script_callbacks.on_ui_tabs(on_ui_tabs)
