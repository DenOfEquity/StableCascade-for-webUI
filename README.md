## Stable Cascade for webui ##
### Forge tested, probably A1111 too ###
I don't think there is anything Forge specific here.
### works for me (tm) on 8Gb VRAM, 16Gb RAM (GTX1070) ###

---
![](screenshot.png "image of extension UI")
---
At your own risk. This is barely tested, and even then only on my computer.
Models will be downloaded automatically, on demand. I only use the 16 bit models. Note that older graphics cards, like mine, don't support bfloat16 and the large prior model is not compatible with normal float16. The smaller model works, but is noticeably worse quality. There is a patched file [here](https://huggingface.co/KBlueLeaf/Stable-Cascade-FP16-fixed/tree/main) which must be put directly into the *models/diffusers* directory. Results are a little different, but it works. Using the fixed version is optional. Using light models is optional.

Image prompt is not image to image, it acts as a style/theme guide.


---
portrait photograph, woman with red hair, wearing green blazer over yellow tshirt and blue trousers, on sunny beach with dark clouds on horizon

![portrait photograph, woman with red hair, wearing green blazer over yellow tshirt and blue trousers, on sunny beach with dark clouds on horizon](example.png "30/10 steps")

---
To do?:
	
controlnet



---
Thanks to:
	[frutiemax92](https://github.com/frutiemax92) for inference_pipeline.py, which helped learn how to put diffusers together
	[benjamin-bertram](https://github.com/benjamin-bertram/sdweb-easy-stablecascade-diffusers) for ui details