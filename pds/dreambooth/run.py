import os
import sys
from pathlib import Path
from pds.utils.trainutil import save_command
import argparse

def main(args):
    model_path = "runwayml/stable-diffusion-v1-5" 
    instance_prompt = args.instance_prompt #"a face of a sks man"
    class_prompt = args.class_prompt #"a face of a man"
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    class_img_dir = save_dir / "class_images"
    instance_dir = args.instance_dir 
    model_output_dir = save_dir
    
    if args.class_dir is None:
        class_img_dir.mkdir(exist_ok=True, parents=True)
    else:
        class_img_dir = args.class_dir
        assert Path(class_img_dir).exists()

    with open(save_dir / "prompts.txt", "w") as f:
        f.write(f"instance_prompt: {instance_prompt}\n")
        f.write(f"class_prompt: {class_prompt}\n")

    #=== save class images ===#
    cmd = ""
    cmd += "accelerate launch train_dreambooth.py "
    cmd += f"--pretrained_model_name_or_path={model_path} "
    cmd += f"--instance_data_dir={instance_dir} "
    cmd += f"--output_dir={model_output_dir} "
    if args.with_prior_preservation:
        cmd += f"--with_prior_preservation --prior_loss_weight=1.0 "
        cmd += f"--class_data_dir={class_img_dir} "
        cmd += f"--class_prompt=\"{class_prompt}\" "
    cmd += f"--instance_prompt=\"{instance_prompt}\" "
    cmd += f"--resolution=512 "
    cmd += f"--train_batch_size=1 "
    cmd += f"--sample_batch_size=1 "
    cmd += f"--gradient_accumulation_steps=1 --gradient_checkpointing "
    cmd += f"--learning_rate={args.learning_rate} "
    cmd += f"--lr_scheduler=\"constant\" "
    cmd += f"--lr_warmup_steps=0 "
    cmd += f"--num_class_images=200 "
    cmd += f"--max_train_steps={args.max_train_steps} "
    cmd += f"--mixed_precision=fp16 "

    print(cmd)
    save_command(save_dir, sys.argv)
    os.system(cmd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_prompt", type=str, help="prompt with special token `sks`")
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--with_prior_preservation", default=True, action="store_true")
    parser.add_argument("--class_prompt",type=str, help="class prompt")
    parser.add_argument("--class_dir", type=str, default=None)
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--instance_dir", type=str, help="few shot example images dir")
    parser.add_argument("--max_train_steps", type=int, default=800)

    args = parser.parse_args()
    main(args)


