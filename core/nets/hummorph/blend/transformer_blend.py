import torch
import torch.nn as nn
import torch.nn.functional as F

class BlendNet(nn.Module):
    def __init__(
        self,
        cfg,
        featmaps_n_channels,
        feat_vol_n_channels,
        mweights_n_channels,
        pos_embed_xyz_dim, 
        pos_embed_dir_dim,
    ):
        super(BlendNet, self).__init__()
        self.cfg = cfg

        self.pix_linear = nn.Sequential(
            nn.Linear(featmaps_n_channels+2*pos_embed_xyz_dim+2*pos_embed_dir_dim+2+mweights_n_channels, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )
        self.volume_linear = nn.Sequential(
            nn.Linear(feat_vol_n_channels+2*pos_embed_xyz_dim+pos_embed_dir_dim+1+mweights_n_channels, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )
        self.query_linear = nn.Sequential(
            nn.Linear(2*pos_embed_xyz_dim+pos_embed_dir_dim+1+mweights_n_channels, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )

        self.encoder = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=4,
            dim_feedforward=64,
            # dim_feedforward=128,
            batch_first=True,
        )
        # self.encoder = nn.TransformerEncoder(
        #     encoder_layer=encoder_layer,
        #     num_layers=2,
        # )

        self.final_attn = nn.MultiheadAttention(
            embed_dim=64,
            num_heads=4,
            add_bias_kv=True,
            add_zero_attn=True,
            batch_first=True,
        )


    def forward(
        self,
        cnl_pts_norm,
        query_pts_with_view_dir,
        feat_volumes,
        rgb_feat,
        pts_with_view_dir,
        **_,
    ):
        point_feat = F.grid_sample(
            input=feat_volumes,
            grid=cnl_pts_norm[None, None, None, :, :],
            padding_mode="zeros",
            align_corners=True,
        )
        point_feat = point_feat[0, :, 0, 0, :].T

        vol_input = torch.cat((point_feat, query_pts_with_view_dir), dim=-1)
        vol_feat = self.volume_linear(vol_input)

        # ? = torch.cat([rgb_feat, torch.zeros(list(rgb_feat.shape[:-1]) + [1]).to(rgb_feat)], dim=-1)
        pix_input = torch.cat((rgb_feat, pts_with_view_dir), dim=-1)
        pix_feat = self.pix_linear(pix_input)

        all_feats = torch.cat((vol_feat.unsqueeze(-2), pix_feat), dim=-2)
        encoder_out = self.encoder(all_feats)

        # what about query depth?
        query = self.query_linear(query_pts_with_view_dir)
        # query = self.query_linear(vol_input)
        attn_out = self.final_attn(
            query.unsqueeze(1), encoder_out, encoder_out, need_weights=False,
        )[0]
        return attn_out.squeeze(1)
