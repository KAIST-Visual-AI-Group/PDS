from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.datamanagers.base_datamanager import \
    VanillaDataManagerConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import \
    NerfstudioDataParserConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from nerfstudio.plugins.types import MethodSpecification
from pds_nerf.modified_nerfstudio.nerfacto import ModifiedNerfactoModelConfig
from pds_nerf.pds_datamanager import PDSDataManagerConfig
from pds_nerf.pds_dataparser import PDSDataParserConfig
from pds_nerf.pds_model import PDSModelConfig
from pds_nerf.pds_pipeline import PDSPipelineConfig
from pds_nerf.pds_trainer import PDSTrainerConfig
from pds_nerf.refinement_pipeline import RefinementPipelineConfig

from pds.pds import PDSConfig

nerfacto_method = MethodSpecification(
    config=TrainerConfig(
        method_name="nerfacto",
        steps_per_eval_batch=500,
        steps_per_eval_all_images=35000,
        steps_per_save=2000,
        max_num_iterations=30000,
        mixed_precision=True,
        experiment_name=None,
        pipeline=VanillaPipelineConfig(
            datamanager=VanillaDataManagerConfig(
                dataparser=NerfstudioDataParserConfig(),
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
                camera_optimizer=CameraOptimizerConfig(
                    mode="SO3xR3",
                    optimizer=AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2),
                    scheduler=ExponentialDecaySchedulerConfig(lr_final=6e-6, max_steps=200000),
                ),
            ),
            model=ModifiedNerfactoModelConfig(eval_num_rays_per_chunk=1 << 15),
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
            },
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="ModifiedNerfacto",
)
pds_method = MethodSpecification(
    config=PDSTrainerConfig(
        method_name="pds",
        steps_per_eval_batch=999999,
        steps_per_eval_image=999999,
        steps_per_eval_all_images=99999999,
        steps_per_save=1000,
        max_num_iterations=30000,
        save_only_latest_checkpoint=False,
        mixed_precision=False,
        load_scheduler=False,
        pipeline=PDSPipelineConfig(
            pds=PDSConfig(src_prompt="", tgt_prompt=""),
            datamanager=PDSDataManagerConfig(
                dataparser=PDSDataParserConfig(),
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
                patch_size=32,
                camera_optimizer=CameraOptimizerConfig(
                    mode="SO3xR3",
                    optimizer=AdamOptimizerConfig(lr=1e-30, eps=1e-8, weight_decay=1e-2),
                ),
            ),
            model=PDSModelConfig(
                eval_num_rays_per_chunk=1 << 15,
                use_appearance_embedding=False,
            ),
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-3, warmup_steps=1000),
            },
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-3, warmup_steps=1000),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="PDS-based NeRF editing method",
)

refinement_method = MethodSpecification(
    config=PDSTrainerConfig(
        method_name="pds_refinement",
        steps_per_eval_batch=999999,
        steps_per_eval_image=999999,
        steps_per_eval_all_images=99999999,
        steps_per_save=1000,
        max_num_iterations=15000,
        save_only_latest_checkpoint=False,
        mixed_precision=False,
        load_scheduler=False,
        pipeline=RefinementPipelineConfig(
            pds=PDSConfig(src_prompt="", tgt_prompt="", num_inference_steps=20, guidance_scale=15),
            skip_min_ratio=0.8,
            skip_max_ratio=0.9,
            datamanager=PDSDataManagerConfig(
                dataparser=PDSDataParserConfig(),
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
                patch_size=32,
                camera_optimizer=CameraOptimizerConfig(
                    mode="SO3xR3",
                    optimizer=AdamOptimizerConfig(lr=1e-30, eps=1e-8, weight_decay=1e-2),
                ),
            ),
            model=PDSModelConfig(
                eval_num_rays_per_chunk=1 << 15,
                use_appearance_embedding=False,
            ),
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-3, warmup_steps=1000),
            },
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-3, warmup_steps=1000),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="Refinement Stage of PDS",
)
