class Get_Correlation(nn.Module):
    def __init__(self, channels, n=3):
        super().__init__()
        self.n = n  # Time difference to consider for each correlation calculation
        reduction_channel = channels // 16
        self.down_conv = nn.Conv3d(
            channels, reduction_channel, kernel_size=1, bias=False)

        self.down_conv2 = nn.Conv3d(
            channels, channels, kernel_size=1, bias=False)
        self.spatial_aggregation = nn.Conv3d(reduction_channel, reduction_channel, kernel_size=(
            9, 3, 3), padding=(4, 1, 1), groups=reduction_channel)
        self.weights = nn.Parameter(
            torch.tensor([0.5, 0.5]), requires_grad=True)
        self.conv_back = nn.Conv3d(
            reduction_channel, channels, kernel_size=1, bias=False)

    def forward(self, x):
        B, C, T, H, W = x.size()
        # padding = torch.zeros(B, C, self.n, H, W).to(x.device)
        x_padded = torch.cat([x[:, :, :self.n].repeat(
            1, 1, self.n, 1, 1), x, x[:, :, -self.n:].repeat(1, 1, self.n, 1, 1)], dim=2)

        # Compute correlation for t-n frame
        x2_prev = self.down_conv2(x_padded[:, :, self.n-1:T+self.n-1, :])
        affinities_prev = torch.einsum('bcthw,bctsd->bthwsd', x, x2_prev)
        features_prev = torch.einsum(
            'bctsd,bthwsd->bcthw', x2_prev, F.sigmoid(affinities_prev)-0.5) * self.weights[0]

        # Compute correlation for t+n frame
        x2_next = self.down_conv2(x_padded[:, :, self.n+1:T+self.n+1, :])
        affinities_next = torch.einsum('bcthw,bctsd->bthwsd', x, x2_next)
        features_next = torch.einsum(
            'bctsd,bthwsd->bcthw', x2_next, F.sigmoid(affinities_next)-0.5) * self.weights[1]

        features = features_prev + features_next

        x = self.down_conv(x)
        aggregated_x = self.spatial_aggregation(x)
        aggregated_x = self.conv_back(aggregated_x)

        return features * (F.sigmoid(aggregated_x)-0.5)
