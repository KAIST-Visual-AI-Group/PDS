from pds_nerf.data.datamanagers.pds_datamanager import PDSDataManagerConfig
from pds_nerf.data.datamanagers.pds_splat_datamanager import \
    PDSSplatDataManagerConfig
from pds_nerf.data.dataparsers.pds_dataparser import PDSDataParserConfig
from pds_nerf.engine.pds_trainer import PDSTrainerConfig
from pds_nerf.models.pds_nerfacto import PDSNerfactoModelConfig
from pds_nerf.models.pds_splatfacto import PDSSplatfactoModelConfig
from pds_nerf.pipelines.pds_pipeline import PDSPipelineConfig
from pds_nerf.pipelines.refinement_pipeline import RefinementPipelineConfig

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
            model=PDSNerfactoModelConfig(eval_num_rays_per_chunk=1 << 15),
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
    description="Nerfacto that can turn off the use of appearance embedding",
)
pds_method = MethodSpecification(
    config=PDSTrainerConfig(
        method_name="pds",
        steps_per_eval_batch=999999,
        steps_per_eval_image=999999,
        steps_per_eval_all_images=99999999,
        steps_per_save=1000,
        max_num_iterations=30000,
        save_only_latest_checkpoint=True,
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
            model=PDSNerfactoModelConfig(
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
        save_only_latest_checkpoint=True,
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
            model=PDSNerfactoModelConfig(
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

pds_splat_method = MethodSpecification(
    config=PDSTrainerConfig(
        method_name="pds_splat",
        steps_per_eval_batch=999999,
        steps_per_eval_image=999999,
        steps_per_eval_all_images=99999999,
        steps_per_save=1000,
        max_num_iterations=30000,
        save_only_latest_checkpoint=True,
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
            model=PDSSplatfactoModelConfig(
                eval_num_rays_per_chunk=1 << 15,
                num_downscales=0,
            ),
        ),
        optimizers={
            "xyz": {
                "optimizer": AdamOptimizerConfig(lr=1.6e-4, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1.6e-6,
                    max_steps=30000,
                ),
            },
            "features_dc": {
                "optimizer": AdamOptimizerConfig(lr=0.0025, eps=1e-15),
                "scheduler": None,
            },
            "features_rest": {
                "optimizer": AdamOptimizerConfig(lr=0.0025 / 20, eps=1e-15),
                "scheduler": None,
            },
            "opacity": {
                "optimizer": AdamOptimizerConfig(lr=0.05, eps=1e-15),
                "scheduler": None,
            },
            "scaling": {
                "optimizer": AdamOptimizerConfig(lr=0.005, eps=1e-15),
                "scheduler": None,
            },
            "rotation": {"optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15), "scheduler": None},
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=5e-5, max_steps=30000),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="PDS-based 3D Gaussian Splat editing method",
)

pds_splat_refinement_method = MethodSpecification(
    config=PDSTrainerConfig(
        method_name="pds_splat_refinement",
        steps_per_eval_batch=999999,
        steps_per_eval_image=999999,
        steps_per_eval_all_images=99999999,
        steps_per_save=1000,
        max_num_iterations=30000,
        save_only_latest_checkpoint=True,
        mixed_precision=False,
        load_scheduler=False,
        pipeline=RefinementPipelineConfig(
            pds=PDSConfig(src_prompt="", tgt_prompt="", num_inference_steps=20, guidance_scale=15),
            skip_min_ratio=0.8,
            skip_max_ratio=0.9,
            datamanager=PDSSplatDataManagerConfig(
                dataparser=PDSDataParserConfig(),
                patch_size=32,
            ),
            model=PDSSplatfactoModelConfig(
                num_downscales=0,
                stop_split_at=0,  # juil: Splitting GS in the refinement stage is not implemented yet.
                eval_num_rays_per_chunk=1 << 15,
            ),
        ),
        optimizers={
            "xyz": {
                "optimizer": AdamOptimizerConfig(lr=1.6e-4, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1.6e-6,
                    max_steps=30000,
                ),
            },
            "features_dc": {
                "optimizer": AdamOptimizerConfig(lr=0.0025, eps=1e-15),
                "scheduler": None,
            },
            "features_rest": {
                "optimizer": AdamOptimizerConfig(lr=0.0025 / 20, eps=1e-15),
                "scheduler": None,
            },
            "opacity": {
                "optimizer": AdamOptimizerConfig(lr=0.05, eps=1e-15),
                "scheduler": None,
            },
            "scaling": {
                "optimizer": AdamOptimizerConfig(lr=0.005, eps=1e-15),
                "scheduler": None,
            },
            "rotation": {"optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15), "scheduler": None},
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=5e-5, max_steps=30000),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="Refinement Stage of PDS-Splat",
)
