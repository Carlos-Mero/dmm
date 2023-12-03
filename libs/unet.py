import torch
from torch import nn
from helper import *

class Unet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        in_channels = config.model.in_channels
        out_channels = config.model.out_channels
        ch_mults = config.model.latent_channel_multipilers
        attn_resolutions = config.model.attn_resolutions
        num_resolutions = len(config.model.latent_channel_multipilers)
        dropout = config.model.dropout
        resamp_with_conv = config.model.resamp_with_conv
        self.latent_channels = config.model.latent_channels
        self.time_embedding_channels = self.latent_channels * 4
        self.num_res_blocks = config.model.num_res_blocks
        # time step embedding
        self.temb = nn.Module()
        self.temb_dense = nn.ModuleList([
            nn.Linear(self.latent_channels, self.time_embedding_channels),
            nn.Linear(self.time_embedding_channels, self.time_embedding_channels)
        ])
        # down sampling
        #------------------------------------------------------------------------
        self.conv_in = nn.Conv2d(in_channels=in_channels,
                                 out_channels=self.latent_channels,
                                 kernel_size=3,
                                 padding=1,
                                 stride=1)
        curr_resolution = config.model.resolution
        in_ch_mults = (1,) + ch_mults
        self.down = nn.ModuleList()
        block_in = None
        for i_level in range(num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = self.latent_channels * in_ch_mults[i_level]
            block_out = self.latent_channels * ch_mults[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_resolution in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions-1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        #------------------------------------------------------------------------
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.time_embedding_channels,
                                       dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.time_embedding_channels,
                                       dropout=dropout)
        # upsampling
        #------------------------------------------------------------------------
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = self.latent_channels * ch_mults[i_level]
            skip_in = self.latent_channels * ch_mults[i_level]
            for i_block in range(self.num_res_blocks+1):
                if i_block == self.num_res_blocks:
                    skip_in = self.latent_channels * in_ch_mults[i_level]
                block.append(ResnetBlock(in_channels=block_in+skip_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order
        # end
        #------------------------------------------------------------------------
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(in_channels=block_in,
                                        out_channels=out_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x, t):
        assert x.shape[2] == x.shape[3] == self.resolution

        # timestep embedding
        #------------------------------------------------------------------------
        temb = get_timestep_embedding(t, self.ch)
        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.temb.dense[1](temb)

        # downsampling
        #------------------------------------------------------------------------
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions-1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        #------------------------------------------------------------------------
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        #------------------------------------------------------------------------
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](
                    torch.cat([h, hs.pop()], dim=1), temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        #------------------------------------------------------------------------
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h
