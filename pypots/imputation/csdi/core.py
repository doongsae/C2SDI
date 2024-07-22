"""
The core wrapper assembles the submodules of CSDI imputation model
and takes over the forward progress of the algorithm.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import torch
import torch.nn as nn

from ...nn.modules.csdi import BackboneCSDI


class _CSDI(nn.Module):
    def __init__(
        self,
        n_features,
        n_layers,
        n_heads,
        n_channels,
        d_time_embedding,
        d_feature_embedding,
        d_diffusion_embedding,
        is_unconditional,
        n_diffusion_steps,
        schedule,
        beta_start,
        beta_end,

        # ! EDIT
        d_class_embedding,
        n_classes,

        # * Algorithm 2
        w
    ):
        super().__init__()

        self.n_features = n_features
        self.d_time_embedding = d_time_embedding
        self.is_unconditional = is_unconditional

        # ! EDIT
        self.d_class_embedding = d_class_embedding
        self.n_classes = n_classes

        self.embed_layer = nn.Embedding(
            num_embeddings=n_features,
            embedding_dim=d_feature_embedding,
        )

        # ! EDIT
        self.class_embed_layer = nn.Embedding(
            num_embeddings=n_classes,
            embedding_dim=d_class_embedding,
        )

        self.backbone = BackboneCSDI(
            n_layers,
            n_heads,
            n_channels,
            n_features,
            d_time_embedding,
            d_feature_embedding,
            d_diffusion_embedding,
            is_unconditional,
            n_diffusion_steps,
            schedule,
            beta_start,
            beta_end,

            # ! EDIT
            d_class_embedding,
            n_classes,

            # * Algorithm 2
            w
        )

    @staticmethod
    def time_embedding(pos, d_model=128):
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model).to(pos.device)
        position = pos.unsqueeze(2)
        div_term = 1 / torch.pow(
            10000.0, torch.arange(0, d_model, 2, device=pos.device) / d_model
        )
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe

    # ! EDIT: 마지막 parameter로 class_label 추가
    def get_side_info(self, observed_tp, cond_mask, class_label):
        B, K, L = cond_mask.shape
        device = observed_tp.device
        time_embed = self.time_embedding(
            observed_tp, self.d_time_embedding
        )  # (B,L,emb)
        time_embed = time_embed.to(device)
        time_embed = time_embed.unsqueeze(2).expand(-1, -1, K, -1)
        feature_embed = self.embed_layer(
            torch.arange(self.n_features).to(device)
        )  # (K,emb)
        feature_embed = feature_embed.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)

        # ! EDIT
        class_embed = self.class_embed_layer(class_label)
        class_embed = class_embed.unsqueeze(1).unsqueeze(2).expand(-1, L, K, -1)

        # ! EDIT: torch.cat에 class_embed 추가
        side_info = torch.cat(
            [time_embed, feature_embed, class_embed], dim=-1
        )  # (B,L,K,emb+d_feature_embedding)
        side_info = side_info.permute(0, 3, 2, 1)  # (B,*,K,L)

        if not self.is_unconditional:
            side_mask = cond_mask.unsqueeze(1)  # (B,1,K,L)
            side_info = torch.cat([side_info, side_mask], dim=1)

        return side_info

    # ! EDIT: 모든 parameter들에 class_label 추가
    def forward(self, inputs, training=True, n_sampling_times=1):
        results = {}
        if training:  # for training
            (observed_data, indicating_mask, cond_mask, observed_tp, class_label) = (
                inputs["X_ori"],
                inputs["indicating_mask"],
                inputs["cond_mask"],
                inputs["observed_tp"],
                inputs["class_label"]
            )
            side_info = self.get_side_info(observed_tp, cond_mask, class_label)
            
            # * Algorithm 2: class_label 추가, observed_tp 전달
            training_loss = self.backbone.calc_loss(
                observed_data, cond_mask, indicating_mask, side_info, training, observed_tp, class_label
            )
            results["loss"] = training_loss
        elif not training and n_sampling_times == 0:  # for validating
            (observed_data, indicating_mask, cond_mask, observed_tp, class_label) = (
                inputs["X_ori"],
                inputs["indicating_mask"],
                inputs["cond_mask"],
                inputs["observed_tp"],
                inputs["class_label"]
            )
            side_info = self.get_side_info(observed_tp, cond_mask, class_label)
            
            # * Algorithm 2: class_label 추가, observed_tp 전달
            validating_loss = self.backbone.calc_loss_valid(
                observed_data, cond_mask, indicating_mask, side_info, training, observed_tp, class_label
            )
            results["loss"] = validating_loss
        elif not training and n_sampling_times > 0:  # for testing
            # print("Testing core!")
            observed_data, cond_mask, observed_tp, class_label = (
                inputs["X"],
                inputs["cond_mask"],
                inputs["observed_tp"],
                inputs["class_label"]
            )
            side_info = self.get_side_info(observed_tp, cond_mask, class_label)
            
            # * Algorithm 2: class_label 추가, observed_tp 전달
            samples = self.backbone(
                observed_data, cond_mask, side_info, n_sampling_times, observed_tp, class_label
            )  # (n_samples, n_sampling_times, n_features, n_steps)
            repeated_obs = observed_data.unsqueeze(1).repeat(1, n_sampling_times, 1, 1)
            repeated_mask = cond_mask.unsqueeze(1).repeat(1, n_sampling_times, 1, 1)
            imputed_data = repeated_obs + samples * (1 - repeated_mask)

            results["imputed_data"] = imputed_data.permute(
                0, 1, 3, 2
            )  # (n_samples, n_sampling_times, n_steps, n_features)

        return results
