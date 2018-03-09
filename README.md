# padless_cnn
Filter size planning for "padless" deep CNN design

The goal is to avoid padding (adding artificial zeros to) the feature maps in a deep convolution network. To accomplish this, filter sizes should be carefully designed.

I take a brute-force approach in computing the compatible filter and input sizes to reach a certain depth in a CNN. 
