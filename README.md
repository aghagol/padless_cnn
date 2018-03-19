# padless_cnn
Filter size planning for "padless" deep CNN design

Using `Python 3`

The goal is to avoid padding, i.e. extending the width and height of the feature maps by appending rows and columns of zeros to the tensor borders, in a deep convolution network. To accomplish this, filter sizes should be carefully designed.

Here, we take a brute-force approach in computing the compatible filter and input sizes to reach a certain depth in a CNN.

Example output for U-Net with target input image shape `{"height":512, "width":512}`, filter choices of sizes 3, 4, 5, U-Net depth from 5 to 10 and stride of 2 in all layers.

Note: Input sizes are reported for all designs with `starting_size` (feature map size in the deepest layer) in the range from 1 to 9, resulting in 9 input sizes being reported for each filter configuration. So, a network that can accept images with height and width 512 can also accept images with reported sizes.

```
Generating filter sizes for height:
filter size sequence (deep to shallow): [3, 3, 3, 3, 3, 3, 3, 4], proper input sizes: [512, 768, 1024, 1280, 1536, 1792, 2048, 2304, 2560]
filter size sequence (deep to shallow): [4, 4, 4, 4, 4, 5, 4], proper input sizes: [384, 512, 640, 768, 896, 1024, 1152, 1280, 1408]
filter size sequence (deep to shallow): [4, 4, 4, 4, 5, 3, 4], proper input sizes: [384, 512, 640, 768, 896, 1024, 1152, 1280, 1408]
filter size sequence (deep to shallow): [4, 4, 4, 5, 3, 3, 4], proper input sizes: [384, 512, 640, 768, 896, 1024, 1152, 1280, 1408]
filter size sequence (deep to shallow): [4, 4, 5, 3, 3, 3, 4], proper input sizes: [384, 512, 640, 768, 896, 1024, 1152, 1280, 1408]
filter size sequence (deep to shallow): [4, 5, 3, 3, 3, 3, 4], proper input sizes: [384, 512, 640, 768, 896, 1024, 1152, 1280, 1408]
filter size sequence (deep to shallow): [5, 3, 3, 3, 3, 3, 4], proper input sizes: [384, 512, 640, 768, 896, 1024, 1152, 1280, 1408]
filter size sequence (deep to shallow): [3, 3, 3, 3, 3, 3, 4], proper input sizes: [256, 384, 512, 640, 768, 896, 1024, 1152, 1280]
filter size sequence (deep to shallow): [4, 4, 4, 4, 5, 4], proper input sizes: [192, 256, 320, 384, 448, 512, 576, 640, 704]
filter size sequence (deep to shallow): [4, 4, 4, 5, 3, 4], proper input sizes: [192, 256, 320, 384, 448, 512, 576, 640, 704]
filter size sequence (deep to shallow): [4, 4, 5, 3, 3, 4], proper input sizes: [192, 256, 320, 384, 448, 512, 576, 640, 704]
filter size sequence (deep to shallow): [4, 5, 3, 3, 3, 4], proper input sizes: [192, 256, 320, 384, 448, 512, 576, 640, 704]
filter size sequence (deep to shallow): [5, 3, 3, 3, 3, 4], proper input sizes: [192, 256, 320, 384, 448, 512, 576, 640, 704]
filter size sequence (deep to shallow): [3, 3, 3, 3, 3, 4], proper input sizes: [128, 192, 256, 320, 384, 448, 512, 576, 640]
Generating filter sizes for width:
filter size sequence (deep to shallow): [3, 3, 3, 3, 3, 3, 3, 4], proper input sizes: [512, 768, 1024, 1280, 1536, 1792, 2048, 2304, 2560]
filter size sequence (deep to shallow): [4, 4, 4, 4, 4, 5, 4], proper input sizes: [384, 512, 640, 768, 896, 1024, 1152, 1280, 1408]
filter size sequence (deep to shallow): [4, 4, 4, 4, 5, 3, 4], proper input sizes: [384, 512, 640, 768, 896, 1024, 1152, 1280, 1408]
filter size sequence (deep to shallow): [4, 4, 4, 5, 3, 3, 4], proper input sizes: [384, 512, 640, 768, 896, 1024, 1152, 1280, 1408]
filter size sequence (deep to shallow): [4, 4, 5, 3, 3, 3, 4], proper input sizes: [384, 512, 640, 768, 896, 1024, 1152, 1280, 1408]
filter size sequence (deep to shallow): [4, 5, 3, 3, 3, 3, 4], proper input sizes: [384, 512, 640, 768, 896, 1024, 1152, 1280, 1408]
filter size sequence (deep to shallow): [5, 3, 3, 3, 3, 3, 4], proper input sizes: [384, 512, 640, 768, 896, 1024, 1152, 1280, 1408]
filter size sequence (deep to shallow): [3, 3, 3, 3, 3, 3, 4], proper input sizes: [256, 384, 512, 640, 768, 896, 1024, 1152, 1280]
filter size sequence (deep to shallow): [4, 4, 4, 4, 5, 4], proper input sizes: [192, 256, 320, 384, 448, 512, 576, 640, 704]
filter size sequence (deep to shallow): [4, 4, 4, 5, 3, 4], proper input sizes: [192, 256, 320, 384, 448, 512, 576, 640, 704]
filter size sequence (deep to shallow): [4, 4, 5, 3, 3, 4], proper input sizes: [192, 256, 320, 384, 448, 512, 576, 640, 704]
filter size sequence (deep to shallow): [4, 5, 3, 3, 3, 4], proper input sizes: [192, 256, 320, 384, 448, 512, 576, 640, 704]
filter size sequence (deep to shallow): [5, 3, 3, 3, 3, 4], proper input sizes: [192, 256, 320, 384, 448, 512, 576, 640, 704]
filter size sequence (deep to shallow): [3, 3, 3, 3, 3, 4], proper input sizes: [128, 192, 256, 320, 384, 448, 512, 576, 640]
```
