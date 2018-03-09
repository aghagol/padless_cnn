# padless_cnn
Filter size planning for "padless" deep CNN design

Using `Python 3`

The goal is to avoid padding (adding artificial zeros to) the feature maps in a deep convolution network. To accomplish this, filter sizes should be carefully designed.

I take a brute-force approach in computing the compatible filter and input sizes to reach a certain depth in a CNN.

Example output for U-Net with target input image shape `{"height":100, "width":200}`, filter choices of sizes 2, 3, 4, 5, 6, U-Net depth from 3 to 9 and stride of 2 in all layers.

```
Generating filter sizes for height:
filter size sequence (deep to shallow): [3, 5, 5, 6, 6], input size: 96
filter size sequence (deep to shallow): [3, 5, 6, 4, 6], input size: 96
filter size sequence (deep to shallow): [3, 5, 6, 5, 4], input size: 96
filter size sequence (deep to shallow): [3, 5, 6, 5, 5], input size: 97
filter size sequence (deep to shallow): [3, 5, 6, 5, 6], input size: 98
....
filter size sequence (deep to shallow): [6, 2, 2, 2, 2], input size: 96
filter size sequence (deep to shallow): [6, 2, 2, 2, 3], input size: 97
filter size sequence (deep to shallow): [6, 2, 2, 2, 4], input size: 98
filter size sequence (deep to shallow): [6, 2, 2, 2, 5], input size: 99
filter size sequence (deep to shallow): [6, 2, 2, 3, 2], input size: 98
filter size sequence (deep to shallow): [2, 2, 3, 5, 6, 6], input size: 96
filter size sequence (deep to shallow): [2, 2, 3, 6, 4, 6], input size: 96
filter size sequence (deep to shallow): [2, 2, 3, 6, 5, 4], input size: 96
filter size sequence (deep to shallow): [2, 2, 3, 6, 5, 5], input size: 97
filter size sequence (deep to shallow): [2, 2, 3, 6, 5, 6], input size: 98
...
filter size sequence (deep to shallow): [2, 4, 2, 2, 3, 3], input size: 99
filter size sequence (deep to shallow): [3, 2, 2, 2, 2, 2], input size: 96
filter size sequence (deep to shallow): [3, 2, 2, 2, 2, 3], input size: 97
filter size sequence (deep to shallow): [3, 2, 2, 2, 2, 4], input size: 98
filter size sequence (deep to shallow): [3, 2, 2, 2, 2, 5], input size: 99
Generating filter sizes for width:
filter size sequence (deep to shallow): [3, 5, 5, 6, 6, 6], input size: 196
filter size sequence (deep to shallow): [3, 5, 6, 4, 6, 6], input size: 196
filter size sequence (deep to shallow): [3, 5, 6, 5, 4, 6], input size: 196
filter size sequence (deep to shallow): [3, 5, 6, 5, 5, 4], input size: 196
filter size sequence (deep to shallow): [3, 5, 6, 5, 5, 5], input size: 197
...
filter size sequence (deep to shallow): [6, 2, 2, 2, 5, 3], input size: 199
filter size sequence (deep to shallow): [6, 2, 2, 3, 2, 2], input size: 196
filter size sequence (deep to shallow): [6, 2, 2, 3, 2, 3], input size: 197
filter size sequence (deep to shallow): [6, 2, 2, 3, 2, 4], input size: 198
filter size sequence (deep to shallow): [6, 2, 2, 3, 2, 5], input size: 199
filter size sequence (deep to shallow): [2, 2, 3, 5, 6, 6, 6], input size: 196
filter size sequence (deep to shallow): [2, 2, 3, 6, 4, 6, 6], input size: 196
filter size sequence (deep to shallow): [2, 2, 3, 6, 5, 4, 6], input size: 196
filter size sequence (deep to shallow): [2, 2, 3, 6, 5, 5, 4], input size: 196
filter size sequence (deep to shallow): [2, 2, 3, 6, 5, 5, 5], input size: 197
...
filter size sequence (deep to shallow): [3, 2, 2, 2, 2, 5, 3], input size: 199
filter size sequence (deep to shallow): [3, 2, 2, 2, 3, 2, 2], input size: 196
filter size sequence (deep to shallow): [3, 2, 2, 2, 3, 2, 3], input size: 197
filter size sequence (deep to shallow): [3, 2, 2, 2, 3, 2, 4], input size: 198
filter size sequence (deep to shallow): [3, 2, 2, 2, 3, 2, 5], input size: 199
```