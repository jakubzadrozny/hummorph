# adapted from: https://github.com/szymanowiczs/viewset-diffusion/blob/main/model/multi_image_unet.py
# which was adapted from https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py
import torch
import torch.nn as nn

from .unet_parts import (
    ResnetBlock, 
    PreNorm,
    LinearAttention,
    DecoderCrossAttention,
    Residual,
    Downsample,
    Upsample,
)


class UNet3D(nn.Module):
    """
    3D DDPM U-Net which accepts input shaped as
    B x Cond x Channels x Height x Width x Depth
    """
    def __init__(self, cfg, in_channels):
        super(UNet3D, self).__init__()
        
        self.cfg = cfg
        self.use_attention_aggregation = self.cfg.attention_aggregation
        self.blocks_per_res = self.cfg.blocks_per_res

        # input dimensions and initial convolutional layer
        # if self.cfg.model.unet.self_condition and \
        #         not self.cfg.model.feature_extractor_2d.use:
        #     in_channels += 3
        dim = self.cfg.model_channels
        self.init_conv = nn.Conv3d(in_channels, dim, 7, padding = 3)

        # # ========== time embedding ==========
        # time_dim = dim * 4
        # sinu_pos_emb = SinusoidalPosEmb(dim)
        # fourier_dim = dim
        # self.time_mlp = nn.Sequential(
        #     sinu_pos_emb,
        #     nn.Linear(fourier_dim, time_dim),
        #     nn.GELU(),
        #     nn.Linear(time_dim, time_dim)
        # )
        time_dim = None

        # ========== unet channels ==========
        channel_mult = self.cfg.channel_mult
        self.attn_resolutions = self.cfg.attn_resolutions
        dims = [dim, *map(lambda m: dim * m, channel_mult)]
        in_out = list(zip(dims[:-1], dims[1:]))
        # channels dimensions of intermediate feature maps
        self.ft_chans = []
        # spatial dimensions of intermediate feature maps
        current_side = cfg.volume_size
        self.sides = []

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        if self.use_attention_aggregation:
            self.volume_aggregators = []
        num_resolutions = len(in_out)

        # ========== unet layers ==========
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            layers = []
            # resnet blocks
            for b_idx in range(self.blocks_per_res):
                layers.append(ResnetBlock(dim_in, dim_in, time_emb_dim = time_dim))
                self.ft_chans.append(dim_in)
                self.sides.append(current_side)
            # attention
            if current_side in self.attn_resolutions:
                layers.append(Residual(PreNorm(dim_in, LinearAttention(dim_in))))
            # downsampling
            layers.append(Downsample(dim_in, dim_out) if not is_last else nn.Conv3d(dim_in, dim_out, 3, padding = 1))
            current_side = current_side // 2 if not is_last else current_side
            self.downs.append(nn.ModuleList([*layers]))

        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim = time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, LinearAttention(mid_dim)))
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim = time_dim)
        self.ft_chans.append(mid_dim)
        self.sides.append(current_side)

        if self.use_attention_aggregation:
            self.volume_aggregators.append(DecoderCrossAttention(mid_dim, mid_dim, cfg.n_heads,
                                                                 include_query_as_key = False))
            self.query_volume = nn.Parameter(data = torch.rand((mid_dim,
                                                                current_side//2,
                                                                current_side,
                                                                current_side)),
                                                                requires_grad=True)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            layers = []
            for b_idx in range(self.blocks_per_res):
                if self.use_attention_aggregation:
                    self.volume_aggregators.append(DecoderCrossAttention(dim_in, dim_out, cfg.n_heads,
                                                                         include_query_as_key = False))

                layers.append(ResnetBlock(dim_out + dim_in, dim_out, time_emb_dim = time_dim))
            layers.append(Residual(PreNorm(dim_out, LinearAttention(dim_out))))
            layers.append(Upsample(dim_out, dim_in) if not is_last else nn.Conv3d(dim_out, dim_in, 3, padding = 1))
            # layers.append(nn.Conv3d(dim_out, dim_in, 3, padding = 1))
            self.ups.append(nn.ModuleList([*layers]))

        if self.use_attention_aggregation:
            self.volume_aggregators.append(DecoderCrossAttention(dim, dim, cfg.n_heads,
                                                                 include_query_as_key = False))
            self.volume_aggregators = nn.ModuleList(self.volume_aggregators[::-1])

        self.final_res_block = ResnetBlock(dim * 2, dim, time_emb_dim = time_dim)

        # ========== 3D conv upsampler =========
        if self.cfg.volume_size != self.cfg.out_volume_size:
            self.conv_upsampler = nn.Sequential(
                nn.Upsample(scale_factor = 2, mode = 'nearest'),
                ResnetBlock(dim, dim),
                ResnetBlock(dim, dim)
            )
        else:
            self.conv_upsampler = nn.Identity()

        # ========== output layers ==========
        if self.cfg.explicit_volume:
            self.out_color = nn.Conv3d(dim, 3, 1)
            self.out_sigma = nn.Conv3d(dim, 1, 1)

            # initialization of the output layer - see SingleLayerReconstructor for explanation
            fc_gain = (cfg.render.max_depth-cfg.render.min_depth) / cfg.render.n_pts_per_ray
            nn.init.xavier_uniform_(self.out_sigma.weight, fc_gain)
            nn.init.constant_(self.out_sigma.bias, 4.0 * fc_gain)

        self.glob_feat_attn = nn.MultiheadAttention(
            embed_dim=512,
            num_heads=4,
            kdim=mid_dim,
            vdim=mid_dim,
            batch_first=True,
        )
        self.glob_query = nn.Parameter(data=torch.rand((512)), requires_grad=True)

    def forward(self, x, t=None):
        """
        volumes: (B x Cond x C x D x H x W)
        t: (B x Cond)
        """
        B, Cond, C, D, H, W = x.shape
        # encoder_emb = self.time_mlp(t.reshape(B*Cond,))
        # need some options here, for now maxpool the embedding across the set
        # decoder_emb = encoder_emb.reshape(B, Cond, -1).max(dim=1, keepdim=False)[0]
        encoder_emb = None
        decoder_emb = None

        ft_map_idx = 0

        x = self.init_conv(x.reshape(-1, C, D, H, W))
        r = x.reshape(B, Cond, self.ft_chans[ft_map_idx], D, H, W).clone()
        
        h = []
        for down in self.downs:
            res_blocks = down[:self.blocks_per_res]
            if self.sides[ft_map_idx] in self.attn_resolutions:
                attn, downsample = down[self.blocks_per_res:]
            else:
                downsample = down[self.blocks_per_res:][0]
            for r_idx, res_block in enumerate(res_blocks):
                x = res_block(x, encoder_emb)
                if r_idx == self.blocks_per_res - 1 \
                        and self.sides[ft_map_idx] in self.attn_resolutions:
                    x = attn(x)
                curr_dims = x.shape[-3:]
                h.append(x.reshape(B, Cond, self.ft_chans[ft_map_idx], *curr_dims))
                ft_map_idx += 1
            x = downsample(x)

        x = self.mid_block1(x, encoder_emb)
        x = self.mid_attn(x)
        x = self.mid_block2(x, encoder_emb)

        curr_dims = x.shape[-3:]
        x = x.reshape(B, Cond, self.ft_chans[ft_map_idx], *curr_dims)

        ft_map_idx += 1

        x_all = x.permute((0, 1, 3, 4, 5, 2)).reshape(B, -1, self.ft_chans[ft_map_idx-1])
        glob_latent = self.glob_feat_attn(self.glob_query[None, None, :], x_all, x_all, need_weights=False)[0][0]

        if self.use_attention_aggregation:
            # if self.query_volume.shape[-3:] == x.shape[-3:]:
            #     query_vol = self.query_volume
            # else:
            #     query_vol = F.interpolate(self.query_volume.unsqueeze(0), size=x.shape[-3:])
            x = self.volume_aggregators[ft_map_idx](x, self.query_volume)
        else:
            x = torch.mean(x, dim=1, keepdim=False)
        ft_map_idx -= 1

        for up in self.ups:
            res_blocks = up[:self.blocks_per_res]
            attn, upsample = up[self.blocks_per_res:]

            for r_idx, res_block in enumerate(res_blocks):
                scf = h.pop()
                if self.use_attention_aggregation:
                    scf = self.volume_aggregators[ft_map_idx](scf, x)
                else:
                    scf = torch.mean(scf, dim=1, keepdim=False)
                ft_map_idx -= 1
                x = torch.cat((x, scf), dim = 1)
                x = res_block(x, decoder_emb)

            x = attn(x)
            x = upsample(x)
            # up_size = h[-1].shape[-3:] if len(h) > 0 else r.shape[-3:]            
            # x = F.interpolate(x, size=up_size)

        assert ft_map_idx == 0
        if self.use_attention_aggregation:
            x = torch.cat((x, self.volume_aggregators[ft_map_idx](r, x)), dim = 1)
        else:
            x = torch.cat((x, torch.mean(r, dim=1, keepdim=False)), dim = 1)

        x = self.final_res_block(x, decoder_emb)

        x = self.conv_upsampler(x)

        if self.cfg.explicit_volume:
            colors = self.out_color(x)
            sigma = self.out_sigma(x)
        else:
            colors = x
            sigma = torch.empty_like(x[:, :1, ...], device=x.device)

        return sigma, colors, glob_latent
        # return sigma, colors
