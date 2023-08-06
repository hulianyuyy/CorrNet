class Get_Correlation(nn.Module):
    def __init__(self, channels, num_frames=7):
        super().__init__()
        # Number of frames to consider for each correlation calculation
        self.num_frames = num_frames
        reduction_channel = channels // 16
        self.down_conv = nn.Conv3d(
            channels, reduction_channel, kernel_size=1, bias=False)

        self.down_conv2 = nn.Conv3d(
            channels, channels, kernel_size=1, bias=False)
        self.spatial_aggregation = nn.ModuleList([nn.Conv3d(reduction_channel, reduction_channel, kernel_size=(
            9, 3, 3), padding=(4, 1, 1), groups=reduction_channel) for _ in range(num_frames)])
        self.weights = nn.Parameter(torch.ones(
            num_frames) / num_frames, requires_grad=True)
        self.weights2 = nn.Parameter(torch.ones(
            num_frames) / num_frames, requires_grad=True)
        self.conv_back = nn.Conv3d(
            reduction_channel, channels, kernel_size=1, bias=False)

    def forward(self, x):
        B, C, T, H, W = x.size()
        padding = torch.zeros(B, C, self.num_frames, H, W).to(x.device)
        x_padded = torch.cat([padding, x, padding], dim=2)

        features = 0
        for i in range(self.num_frames):
            x2 = self.down_conv2(x_padded[:, :, i:T+i, :])
            affinities = torch.einsum('bcthw,bctsd->bthwsd', x, x2)
            features += torch.einsum('bctsd,bthwsd->bcthw',
                                     x2, F.sigmoid(affinities)-0.5) * self.weights2[i]

        x = self.down_conv(x)
        aggregated_x = 0
        for i in range(self.num_frames):
            aggregated_x += self.spatial_aggregation[i](x) * self.weights[i]

        aggregated_x = self.conv_back(aggregated_x)

        return features * (F.sigmoid(aggregated_x)-0.5)
