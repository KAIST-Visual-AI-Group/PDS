## NeRF Editing

### Installation
Our NeRF editing code is based on [nerfstudio](https://docs.nerf.studio/).

We have tested our code with torch 2.0.1+cu118, tinycudann 1.7, nerfstudio 0.3.4, diffusers 0.17.1 and transformers 4.30.2.

Install `pds_nerf` library:
```
cd PDS/pds/nerf
pip install -e .
```

You should be able to see `pds` and `pds_refinement` options in the following command:
```
ns-train -h
```

### Finetuning Stable Diffusion with Dreambooth
For NeRF editing, we have found that finetuning Stable Diffusion using [Dreambooth](https://dreambooth.github.io/) effectively reduces discrepancies between input prompts and real image data.

Under the directory `PDS/pds/dreambooth`, you can finetune a pre-trained stable diffusion model with Dreambooth by:
```
python run.py --instance_prompt {"PROMPT_WITH_SPECIAL_TOKEN"} --instance_dir {PATH/TO/DATA_DIR} --class_prompt {"PROMPT_WITHOUT_SPECIAL_TOKEN"} --save_dir {PATH/TO/SAVE_DIR} 
```

An instance prompt describes input images with a special token, for example, "a photo of a sks man", whereas a class prompt is a more general description for generated sample images, such as "a photo of a man", without a special token.

Please refer to the [Dreambooth](https://dreambooth.github.io/) paper or diffuser's [document](https://huggingface.co/docs/diffusers/training/dreambooth) for more details.

### Downloading Image Data and Finetuned Dreambooth Checkpoint
We provide image data and our finetuned Dreambooth checkpoints [here](https://kaistackr-my.sharepoint.com/:f:/g/personal/63days_kaist_ac_kr/EocMB6MBpMJJksILj5_C7TYBsU5MCtKS7Wi8FCjlncLnug?e=TTUJZc).

Below are the instance prompts we used in Dreambooth finetuning:

- person-small: "a photo of a sks man"
- yuseung: "a photo of a sks man"

'person-small' data is from [Instruct-NeRF2NeRF](https://instruct-nerf2nerf.github.io/).

### Run
#### 1. Initialize NeRF
To edit a NeRF scene with PDS, first initiailize a NeRF with Nerfacto:
```
ns-train nerfacto --data {DATA_DIR} --pipeline.model.use_appearance_embedding False
```
Set `pipeline.model.use_appearance_embedding=False`. We have found that not using appearance embedding leads to better editing results, reducing artifacts.

#### 2. Edit NeRF by PDS
Then, now edit the NeRF scene by PDS:
```
ns-train pds --data {DATA_DIR} --load-dir {PATH/TO/NERFACTO_OUTPUT_DIR/nerfstudio_models} \\
		--pipeline.pds.src_prompt {"DESCRIPTION_FOR_ORIGINAL_SCENE"} \\
		--pipeline.pds.tgt_prompt {"DESCRIPTION_FOR_TARGET_SCENE"} \\
		--pipeline.pds.sd_pretrained_model_or_path {PATH/TO/DREAMBOOTH_SAVE_DIR}
```
Here, {DATA\_DIR} should be the same as {DATA\_DIR} used in Nerfacto training. If you use a dreambooth finetuned model as a prior, don't forget to include a special token in both `src_prompt` and `tgt_prompt`.

For instance, you can try editing with our finetuned dreambooth model as follows:
```
ns-train pds --data {.../data/yuseung} --load-dir {PATH/TO/NERFACTO_OUTPUT_DIR/nerfstudio_models} \\
		--pipeline.pds.src_prompt "a photo of a sks man" \\
		--pipeline.pds.tgt_prompt "a photo of a sks Spider Man" \\
		--pipeline.pds.sd_pretrained_model_or_path {.../dreambooth_ckpt/yuseung_dreambooth}
```
It would take around 40GB VRAM with 512x512 images. To run the command within 24GB VRAM, you can use multi GPUs by assigning a different device to a diffusion model: `--pipeline.pds_device "cuda:1"`

After the training, it will automatically render an edited NeRF scene at the same viewpoints of a training dataset and save those rendering images under `PATH/TO/OUTPUT_DIR/eval_outputs`.

#### 3. Refine NeRF by SDEdit
Our full NeRF editing pipeline inlcudes a refinement stage. In the refinement stage, the renderings of an edited NeRF by PDS are updated by [SDEdit](https://sde-image-editing.github.io/) and the NeRF scene is reconstructed with these iteratively updated images.

```
ns-train pds_refinement --data {PATH/TO/PDS_OUTPUT_DIR/eval_outputs} --load-dir {PATH/TO/PDS_OUTPUT_DIR/nerfstudio_models} --pipeline.pds.tgt_prompt {"DESCRIPTIOIN_FOR_TARGET_SCENE"} --pipeline.pds.sd_pretrained_model_or_path {PATH/TO/DREAMBOOTH_SAVE_DIR}
```
Here, `PATH/TO/PDS_OUTPUT_DIR` is the output directory of the previous stage. Pass the same `tgt_prompt` and diffusion model used in the previous stage.


