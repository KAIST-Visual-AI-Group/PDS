# 3D Editing with PDS
Here, we present the code for 3D editing, including [NeRF](https://www.matthewtancik.com/nerf) and [3D Gaussian Splatting (3DGS)](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/), using PDS.


## Installation
Our 3D editing code is based on [nerfstudio](https://docs.nerf.studio/)==1.0.2, diffusers==0.17.1 and transformers==4.30.2.

By running the comands below, install the required dependencies and the `pds_nerf` library:
```
cd PDS/3d_editing
pip install -e .
```

You should be able to see `pds`, `pds_refinement`, `pds_splat`, and `pds_splat_refinement` options in the following command:
```
ns-train -h
```


## Run
### 1. Initialize 3D Scenes
To edit a 3D scene with PDS, first initiailize a scene with Nerfacto or Splatfacto:
```
ns-train nerfacto --data {DATA_DIR} --pipeline.model.use_appearance_embedding False
ns-train splatfacto --data {DATA_DIR}
```
For Nerfacto, it has been found that not using an appearance embedding leads to better results.

### 2. Edit 3D Scenes by PDS
Then, now edit the 3D scene with PDS:
```
ns-train {pds or pds_splat} --data {DATA_DIR} \\
		--load-dir {PATH/TO/OUTPUT_DIR/nerfstudio_models} \\
		--pipeline.pds.src_prompt {"DESCRIPTION_FOR_ORIGINAL_SCENE"} \\
		--pipeline.pds.tgt_prompt {"DESCRIPTION_FOR_TARGET_SCENE"} \\
		--pipeline.pds.sd_pretrained_model_or_path {PATH/TO/MODEL_DIR}
```
Here, {DATA\_DIR} should be the same as {DATA\_DIR} used in the initialization stage. If you use a dreambooth finetuned model as a prior model, don't forget to include a special token in both `src_prompt` and `tgt_prompt`.

Below is an example of an editing command with a finetuned dreambooth model:
```
ns-train {pds or pds_splat} --data {data/yuseung} \\
		--load-dir {PATH/TO/OUTPUT_DIR/nerfstudio_models} \\
		--pipeline.pds.src_prompt "a photo of a sks man" \\
		--pipeline.pds.tgt_prompt "a photo of a sks Spider Man" \\
		--pipeline.pds.sd_pretrained_model_or_path {dreambooth_ckpt/yuseung_dreambooth} \\
		--pipelien.pds.min_step_ratio 0.1 \\
		--pipeline.pds.max_step_ratio 0.9 \\
```
The VRAM requiremrents for `pds` and `pds_splat` are around 40GB and 2GB, respectively, with the resolution of 512x512 images.
To run the command within 24GB VRAM, you can use multiple GPUs by assigning a different device to the prior model: `--pipeline.pds_device "cuda:1"`

After the training, it will automatically render an edited 3D scene at the same viewpoints of a training dataset and save those rendering images under `PATH/TO/OUTPUT_DIR/eval_outputs`.

If the scene doesn't change much from the original scene, you can adjust the t sampling range to [0.1, 0.9] by passing `--pipeline.pds.min_step_ratio 0.1 --pipeline.pds.max_step_ratio 0.9`, which will induce a greater change with a smaller coefficient of the identity preservation term in the PDS equation ($\mathbf{x}_0^{\text{src}} - \mathbf{x}_0^{\text{tgt}}$).

### 3. Refinement Stage
You can refine the 3D scene edited by PDS through a refinement stage.

```
ns-train {pds_refinement or pds_splat_refinement} \\
		--data {PATH/TO/PDS_OUTPUT_DIR/eval_outputs} \\
		--load-dir {PATH/TO/PDS_OUTPUT_DIR/nerfstudio_models} \\
		--pipeline.pds.tgt_prompt {"DESCRIPTION_FOR_TARGET_SCENE"} \\
		--pipeline.pds.sd_pretrained_model_or_path {PATH/TO/MODEL_DIR}
```
Here, `PATH/TO/PDS_OUTPUT_DIR` is the output directory of the previous stage. 
Pass the same `tgt_prompt` and the same diffusion model used in the previous stage.


## Finetuning Stable Diffusion with Dreambooth
For 3D editing, we have found that finetuning Stable Diffusion using [Dreambooth](https://dreambooth.github.io/) effectively reduces discrepancies between input prompts and real image data.

Under the directory `PDS/pds/dreambooth`, you can finetune a pre-trained stable diffusion model with Dreambooth by:
```
python run.py --instance_prompt {"PROMPT_WITH_SPECIAL_TOKEN"} --instance_dir {PATH/TO/DATA_DIR} --class_prompt {"PROMPT_WITHOUT_SPECIAL_TOKEN"} --save_dir {PATH/TO/SAVE_DIR} 
```

An instance prompt describes input images with a special token, for example, "a photo of a __sks__ man", while a class prompt provides a more general description for generated sample images, such as "a photo of a man", without the special token.

Refer to the [Dreambooth](https://dreambooth.github.io/) paper or diffuser's [documentation](https://huggingface.co/docs/diffusers/training/dreambooth) for more details.


## Downloading Image Data and Finetuned Dreambooth Checkpoint
We provide image data and our finetuned Dreambooth checkpoints [here](https://1drv.ms/f/s!AtxL_EOxFeYMk3rftsoc4L8cg0VS?e=Hhbprk).

You can check out the instance prompts used in our Dreambooth finetuning in `prompts.txt` under each checkpoint directory.

A subset of the 3D scene data is from [Instruct-NeRF2NeRF](https://instruct-nerf2nerf.github.io/).
