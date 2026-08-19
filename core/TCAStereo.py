"""Paper-aligned tri-camera extension of the existing SpeckleStereo network."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from core.Speckle_Stereo import SpeckleStereo, autocast
from core.geometry import Combined_Geo_Encoding_Volume
from core.submodule import (
    build_gwc_volume,
    build_gwc_volume_shining,
    context_upsample,
    disparity_regression,
    dispartty_regression_shining,
)
from core.tca import (
    DepthAnythingV2Prior,
    FusionIterationUnit,
    GeometricEnhancementModule,
    GlobalAlignmentModule,
)


class TCAStereo(SpeckleStereo):
    """SpeckleStereo with GEM, calibrated GAM and recurrent FIUs."""

    def __init__(self, args):
        super().__init__(args)
        self.use_gem = bool(getattr(args, "use_gem", True))
        self.use_gam = bool(getattr(args, "use_gam", True))
        self.use_fiu = bool(getattr(args, "use_fiu", True))

        self.gem = GeometricEnhancementModule() if self.use_gem else None
        self.gam = (
            GlobalAlignmentModule(
                ransac_iterations=int(getattr(args, "gam_ransac_iterations", 128)),
                ransac_threshold=float(getattr(args, "gam_ransac_threshold", 0.05)),
                min_points=int(getattr(args, "gam_min_points", 32)),
            )
            if self.use_gam
            else None
        )
        self.fiu = (
            FusionIterationUnit(hidden_channels=int(getattr(args, "fiu_hidden_dim", 32)))
            if self.use_fiu
            else None
        )

        depth_checkpoint = getattr(args, "depth_anything_checkpoint", None)
        self.depth_prior = None
        if depth_checkpoint:
            self.depth_prior = DepthAnythingV2Prior(
                checkpoint_path=depth_checkpoint,
                encoder=getattr(args, "depth_anything_encoder", "vitl"),
                repository_path=getattr(args, "depth_anything_repo", None),
                input_size=int(getattr(args, "depth_anything_input_size", 518)),
            )

    def forward(
        self,
        image1,
        image2,
        texture_image=None,
        calibration=None,
        mono_depth=None,
        iters=12,
        flow_init=None,
        test_mode=False,
        return_details=False,
    ):
        """Estimate disparity from left/right IR and a calibrated texture view.

        ``mono_depth`` may be supplied to use cached Depth Anything V2 output.
        Otherwise ``texture_image`` is evaluated online by the frozen model.
        """

        del flow_init  # Kept in the public signature for base-model compatibility.
        if (self.use_gam or self.use_fiu) and calibration is None:
            raise ValueError("TCAStereo requires intrinsics and an external RT")
        if self.use_gam and mono_depth is None:
            if texture_image is None:
                raise ValueError("TCAStereo requires texture_image or cached mono_depth")
            if self.depth_prior is None:
                raise ValueError(
                    "Depth Anything V2 checkpoint is not loaded; provide mono_depth instead"
                )

        stereo_image_hw = image1.shape[-2:]
        image1 = (2 * (image1 / 255.0) - 1.0).contiguous()
        image2 = (2 * (image2 / 255.0) - 1.0).contiguous()
        gam_result = None
        aligned_depth_left = None
        aligned_depth_valid = None

        with autocast(enabled=self.args.mixed_precision):
            features_left = self.feature(image1)
            features_right = self.feature(image2)
            if self.gem is not None:
                features_left = self.gem(features_left)
                features_right = self.gem(features_right)

            stem_2x = self.stem_2(image1)
            stem_4x = self.stem_4(stem_2x)
            stem_2y = self.stem_2(image2)
            stem_4y = self.stem_4(stem_2y)
            features_left[0] = torch.cat((features_left[0], stem_4x), 1)
            features_right[0] = torch.cat((features_right[0], stem_4y), 1)

            match_left = self.desc(self.conv(features_left[0]))
            match_right = self.desc(self.conv(features_right[0]))
            if self.shining:
                gwc_volume = build_gwc_volume_shining(
                    match_left, match_right, self.args.max_disp // 4, 8
                )
            else:
                gwc_volume = build_gwc_volume(
                    match_left, match_right, self.args.max_disp // 4, 8
                )
            gwc_volume = self.corr_stem(gwc_volume)
            gwc_volume = self.corr_feature_att(gwc_volume, features_left[0])

            channel_correlation_volume = self.mccv(features_left)
            final_correlation_volume = self.cost_agg(
                gwc_volume, channel_correlation_volume
            )
            probability = F.softmax(
                self.classifier(final_correlation_volume).squeeze(1), dim=1
            )
            if self.shining:
                init_disp = dispartty_regression_shining(
                    probability, self.args.max_disp // 4
                )
            else:
                init_disp = disparity_regression(probability, self.args.max_disp // 4)
            del probability, gwc_volume

            if self.gam is not None:
                if mono_depth is None:
                    mono_depth = self.depth_prior(texture_image)
                gam_result = self.gam(
                    init_disp,
                    mono_depth,
                    calibration,
                    stereo_image_hw=stereo_image_hw,
                )
                aligned_depth_left = gam_result.aligned_depth_left
                aligned_depth_valid = gam_result.valid_left

            if not test_mode:
                xspx = self.spx_4(features_left[0])
                xspx = self.spx_2(xspx, stem_2x)
                spx_pred = F.softmax(self.spx(xspx), 1)

            cnet_list = self.cnet(image1, num_layers=self.args.n_gru_layers)
            net_list = [torch.tanh(x[0]) for x in cnet_list]
            inp_list = [torch.relu(x[1]) for x in cnet_list]
            inp_list = [
                list(conv(value).split(split_size=conv.out_channels // 4, dim=1))
                for value, conv in zip(inp_list, self.context_zqr_convs)
            ]

        geo_fn = Combined_Geo_Encoding_Volume(
            match_left.float(),
            match_right.float(),
            final_correlation_volume.float(),
            radius=self.args.corr_radius,
            num_levels=self.args.corr_levels,
        )
        batch, _, height, width = match_left.shape
        coords = (
            torch.arange(width, device=match_left.device, dtype=torch.float32)
            .reshape(1, 1, width, 1)
            .repeat(batch, height, 1, 1)
        )
        disp = init_disp
        disp_preds = []
        fiu_attention = []
        net_cell = None

        for iteration in range(iters):
            disp = disp.detach()
            geo_feature = geo_fn(disp, coords)
            with autocast(enabled=self.args.mixed_precision):
                if self.args.n_gru_layers == 3 and self.args.slow_fast_gru:
                    net_list = self.update_block(
                        net_list,
                        inp_list,
                        iter16=True,
                        iter08=False,
                        iter04=False,
                        update=False,
                    )
                if self.args.n_gru_layers >= 2 and self.args.slow_fast_gru:
                    net_list = self.update_block(
                        net_list,
                        inp_list,
                        iter16=self.args.n_gru_layers == 3,
                        iter08=True,
                        iter04=False,
                        update=False,
                    )
                if net_cell is None:
                    net_cell = net_list
                net_cell, net_list, mask_feat_4, stereo_residual = self.update_block(
                    net_cell,
                    net_list,
                    inp_list,
                    geo_feature,
                    disp,
                    iter16=self.args.n_gru_layers == 3,
                    iter08=self.args.n_gru_layers >= 2,
                )
                if self.fiu is not None:
                    residual, attention = self.fiu(
                        aligned_depth_left,
                        disp,
                        stereo_residual,
                        aligned_depth_valid,
                    )
                    if return_details:
                        fiu_attention.append(attention)
                else:
                    residual = stereo_residual
            disp = disp + residual
            if test_mode and iteration < iters - 1:
                continue
            disp_up = self.upsample_disp(disp, mask_feat_4, stem_2x)
            disp_preds.append(disp_up)

        if test_mode:
            if not return_details:
                return disp_up
            output = {
                "disparity": disp_up,
                "initial_disparity_low": init_disp,
                "monocular_depth_texture": mono_depth,
                "fiu_attention": fiu_attention[-1] if fiu_attention else None,
            }
            if gam_result is not None:
                output.update(
                    {
                        "aligned_depth_left": gam_result.aligned_depth_left,
                        "aligned_depth_texture": gam_result.aligned_depth_texture,
                        "gam_valid_left": gam_result.valid_left,
                        "gam_valid_texture": gam_result.valid_texture,
                        "gam_stereo_depth_texture": gam_result.stereo_depth_texture,
                        "gam_alpha": gam_result.alpha,
                        "gam_beta": gam_result.beta,
                    }
                )
            return output

        init_disp_up = context_upsample(init_disp * 4, spx_pred.float()).unsqueeze(1)
        if return_details:
            return init_disp_up, disp_preds, {
                "monocular_depth_texture": mono_depth,
                "fiu_attention": fiu_attention,
                "gam": gam_result,
            }
        return init_disp_up, disp_preds
