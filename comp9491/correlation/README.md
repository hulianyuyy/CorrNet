## Modify Correlation Module

1. [aggreagatedFrames.py]('./aggreagatedFrames.py'): set a parameter `num_frames` which means aggregate `num_frames` frames to calculate the correlation
2. [scaledDotProduct.py]('./scaledDotProduct.py'): use scaled dot product to calculate correlation instead of dot product
3. [skipFrame.py]('./skipFrame.py'): set a parameter `n` which means skip `n` frames to calculate the correlation

## How to use

Use the code from each file to replace the corresponding `Get_Correlation` code in the original code from [modules/resnet.py](../../modules/resnet.py)
