import os.path as osp

import numpy as np
import torch

from fluxrt import LIVEPORTRAIT_AVAILABLE
from fluxrt.stream_processor.postprocessors.base import BasePostProcessor

if LIVEPORTRAIT_AVAILABLE:
    from liveportrait.config.inference_config import InferenceConfig
    from liveportrait.config.crop_config import CropConfig
    from liveportrait.live_portrait_wrapper import LivePortraitWrapper
    from liveportrait.utils.cropper import Cropper
    from liveportrait.utils.camera import get_rotation_matrix
    from liveportrait.utils.crop import prepare_paste_back, paste_back

_LIP_INDICES = [6, 12, 14, 17, 19, 20]


class LivePortraitPostProcessor(BasePostProcessor):
    def __init__(self, models_dir: str):
        if not LIVEPORTRAIT_AVAILABLE:
            raise RuntimeError("LivePortrait not installed")
        insightface_dir = osp.join(osp.dirname(models_dir), 'insightface')
        landmark_path = osp.join(models_dir, 'landmark.onnx')
        inf_cfg = InferenceConfig(
            checkpoint_F=osp.join(models_dir, 'base_models', 'appearance_feature_extractor.pth'),
            checkpoint_M=osp.join(models_dir, 'base_models', 'motion_extractor.pth'),
            checkpoint_G=osp.join(models_dir, 'base_models', 'spade_generator.pth'),
            checkpoint_W=osp.join(models_dir, 'base_models', 'warping_module.pth'),
            checkpoint_S=osp.join(models_dir, 'retargeting_models', 'stitching_retargeting_module.pth'),
            flag_use_half_precision=True,
            flag_pasteback=True,
            flag_do_crop=True,
            flag_stitching=True,
            flag_relative_motion=True,
        )
        crop_cfg = CropConfig(
            insightface_root=insightface_dir,
            landmark_ckpt_path=landmark_path,
        )
        self.wrapper = LivePortraitWrapper(inf_cfg)
        self.cropper = Cropper(crop_cfg=crop_cfg)
        self.inf_cfg = inf_cfg
        self.crop_cfg = crop_cfg

        import liveportrait.utils.cropper as _lp_cropper
        _lp_cropper.log = lambda *a, **kw: None

    def _get_kp_info(self, rgb: np.ndarray) -> dict | None:
        crop_info = self.cropper.crop_source_image(rgb, self.crop_cfg)
        if crop_info is None:
            return None
        I = self.wrapper.prepare_source(crop_info['img_crop_256x256'])
        return self.wrapper.get_kp_info(I)

    def process(self, source_rgb: np.ndarray, driving_rgb: np.ndarray) -> np.ndarray:
        driving_kp_info = self._get_kp_info(driving_rgb)
        if driving_kp_info is None:
            return source_rgb

        crop_info = self.cropper.crop_source_image(source_rgb, self.crop_cfg)
        if crop_info is None:
            return source_rgb

        I_s = self.wrapper.prepare_source(crop_info['img_crop_256x256'])
        x_s_info = self.wrapper.get_kp_info(I_s)
        x_c_s = x_s_info['kp']
        R_s = get_rotation_matrix(x_s_info['pitch'], x_s_info['yaw'], x_s_info['roll'])
        f_s = self.wrapper.extract_feature_3d(I_s)
        x_s = self.wrapper.transform_keypoint(x_s_info)

        lip_array = torch.from_numpy(self.inf_cfg.lip_array).to(dtype=torch.float32, device=self.wrapper.device)
        delta_new = x_s_info['exp'].clone()
        for idx in _LIP_INDICES:
            delta_new[:, idx, :] = (x_s_info['exp'] + (driving_kp_info['exp'] - lip_array))[:, idx, :]

        t_new = x_s_info['t'].clone()
        t_new[..., 2].fill_(0)
        x_d_new = x_s_info['scale'] * (x_c_s @ R_s + delta_new) + t_new
        x_d_new = self.wrapper.stitching(x_s, x_d_new)
        x_d_new = x_s + (x_d_new - x_s) * self.inf_cfg.driving_multiplier

        out = self.wrapper.warp_decode(f_s, x_s, x_d_new)
        I_p = self.wrapper.parse_output(out['out'])[0]

        mask_ori = prepare_paste_back(
            self.inf_cfg.mask_crop,
            crop_info['M_c2o'],
            dsize=(source_rgb.shape[1], source_rgb.shape[0]),
        )
        return paste_back(I_p, crop_info['M_c2o'], source_rgb, mask_ori)
