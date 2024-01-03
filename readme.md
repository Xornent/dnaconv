```
We trained the multilayered convolution and full-connection network for image 
classification task between 6 categories of symbols: handwritten digits 2, 3, 4, 
6 and handwritten English letters A and B. The Arabic numerals are from the 
corresponding subset of Modified National Institute of Standards and Technology 
(MNIST (Deng, 2012)) dataset, and English alphabets from the extended MNIST 
(EMNIST (Cohen et al., 2017)) dataset. 

Since the original image is 28×28 in both of the dataset, we resized them with 
bilinear interpolation to 8×8 (for digits 2 and 3) and 12×12 (for digits 4 and 6, 
letter A and B). Since the total parameter counts are rather small (a total of 33 
for the network distinguishing 2 and 4, 38 for distinguishing 3 and 6, and 21 
for distinguishing A and B), the size of each training subset in the MNIST/EMNIST 
is sufficient to saturate network performance (about 4000 images per category in 
the training subset). The testing set is the subset of the MNIST/EMNIST test set 
and contains about 800 images per category. These input cells are continuous 
floating point values between [0, 1].

The network differentiating between digits (as in Figure 5) consists of 4 layers. 
Since each input of the layer should be constrained within [0, 1], we scale the 
output by a factor before entering the next layer of network.

$$ A = \frac{ \mathrm{ReLU}(I*K_A-b_A)}{\sum \sum K_A-b_A} $$
$$ B = \frac{ \mathrm{ReLU}(A*K_B-b_B)}{ \sum \sum K_B-b_B } $$
$$ C = \mathrm{ReLU}(W_C B-b_C) $$
$$ D = W_D C - b_D $$

We ensured K_A,b_A,K_B,b_B,W_C,b_C,C,W_D and b_D to be within [0, 1]. K_B is a 
2×2 convolution kernel with stride 2, and in case of 8×8 input, K_A is of size 
2×2 with stride 2. In case of 12×12 input, K_A is of size 3×3 with stride 3. We 
reduce the number of the fully connected layer in the classification task between 
letter A and B.

$$ E = W_E B $$

Since the direct output D or E are not necessarily between 0 and 1, we apply the 
softmax function before the training loss, after multiplying back the scaling 
factor that we have divided previously.

$$ y_{1,2} = D (\sum \sum K_A - b_A )(\sum \sum K_B-b_B) $$

Or in the case of classification of English letters, 

$$ y_{1,2} = E (\sum \sum K_A-b_A )(\sum \sum K_B-b_B) $$
$$ \hat{p_i} = \frac{\exp {y_i}}{\exp{y_1} + \exp{y_2}} $$

We use the sparse categorical cross-entropy loss function

$$ L = \frac{1}{N} \sum_i^N {p_i \log \hat{p_i} +(1-p_i) log (1-\hat{p_i})} $$

The training is performed on PyTorch (Paszke et al., 2019) platform, learning rate 
is set to 0.0001 with stochastic gradient descent (SGD) with weight decay 0.0001 
and momentum 0.5. The network is trained for 100k iterations, and after each 
iteration, the weights and biases are clamped within [0, 1]. The weight matrices 
are initialized by Gaussian distribution with mean 0.5 and standard deviation 0.15. 
The performance of the network on the training and testing set is similar, 
indicating no significant overfitting. All the parameters are given in the 
figures (Figure 5, Figure 7)

[1] Cohen, G., Afshar, S., Tapson, J., & van Schaik, A. (2017). EMNIST: An 
    extension of MNIST to handwritten letters (arXiv:1702.05373). arXiv. 
    https://doi.org/10.48550/arXiv.1702.05373
[2] Deng, L. (2012). The MNIST Database of Handwritten Digit Images for Machine 
    Learning Research [Best of the Web]. IEEE Signal Processing Magazine, 29(6), 
    141–142. https://doi.org/10.1109/MSP.2012.2211477
[3] Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, 
    T., Lin, Z., Gimelshein, N., Antiga, L., Desmaison, A., Köpf, A., Yang, E., 
    DeVito, Z., Raison, M., Tejani, A., Chilamkurthy, S., Steiner, B., Fang, L.,
    ... Chintala, S. (2019). PyTorch: An Imperative Style, High-Performance Deep 
    Learning Library (arXiv:1912.01703). 
    arXiv. https://doi.org/10.48550/arXiv.1912.01703
```