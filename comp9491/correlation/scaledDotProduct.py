class Get_Correlation(nn.Module):
    def __init__(self, channels):
        super().__init__()
        reduction_channel = channels//16
        self.down_conv = nn.Conv3d(
            channels, reduction_channel, kernel_size=1, bias=False)

        self.down_conv2 = nn.Conv3d(
            channels, channels, kernel_size=1, bias=False)
        self.spatial_aggregation1 = nn.Conv3d(reduction_channel, reduction_channel, kernel_size=(
            9, 3, 3), padding=(4, 1, 1), groups=reduction_channel)
        self.spatial_aggregation2 = nn.Conv3d(reduction_channel, reduction_channel, kernel_size=(
            9, 3, 3), padding=(4, 2, 2), dilation=(1, 2, 2), groups=reduction_channel)
        self.spatial_aggregation3 = nn.Conv3d(reduction_channel, reduction_channel, kernel_size=(
            9, 3, 3), padding=(4, 3, 3), dilation=(1, 3, 3), groups=reduction_channel)
        self.weights = nn.Parameter(torch.ones(3) / 3, requires_grad=True)
        self.weights2 = nn.Parameter(torch.ones(2) / 2, requires_grad=True)
        self.conv_back = nn.Conv3d(
            reduction_channel, channels, kernel_size=1, bias=False)

    def forward(self, x):

        x2 = self.down_conv2(x)
        affinities = torch.einsum('bcthw,bctsd->bthwsd', x, torch.concat(
            [x2[:, :, 1:], x2[:, :, -1:]], 2)) / (x2.size(-1) ** 0.5)  # scaled dot-product
        affinities = F.softmax(affinities, dim=-1)  # softmax to get weights

        affinities2 = torch.einsum('bcthw,bctsd->bthwsd', x, torch.concat(
            [x2[:, :, :1], x2[:, :, :-1]], 2)) / (x2.size(-1) ** 0.5)  # scaled dot-product
        affinities2 = F.softmax(affinities2, dim=-1)  # softmax to get weights

        features = torch.einsum('bctsd,bthwsd->bcthw', torch.concat([x2[:, :, 1:], x2[:, :, -1:]], 2), affinities) * self.weights2[0] + \
            torch.einsum('bctsd,bthwsd->bcthw', torch.concat(
                [x2[:, :, :1], x2[:, :, :-1]], 2), affinities2) * self.weights2[1]

        x = self.down_conv(x)
        aggregated_x = self.spatial_aggregation1(x)*self.weights[0] + self.spatial_aggregation2(x)*self.weights[1] \
            + self.spatial_aggregation3(x)*self.weights[2]
        aggregated_x = self.conv_back(aggregated_x)

        return features * (F.sigmoid(aggregated_x)-0.5)
