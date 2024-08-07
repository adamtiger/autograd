# Automatic gradient calculation

This is an example implementation for calculating gradients
for different neural network operations.

The tensor.py contains the details on how the back-propagation on the whole graph happens (based on the chain-rule).

The ops.py contains the operator implementations and the corresponding derivatives.

Below is a set of operations with backward implementations.

## List of operators

- [Gelu](ops.py#L113) ([explanation](#gelu))
- [MatMul](ops.py#L207) ([explanation](#matmul))
- [Convolution](ops.py#L311) ([explanation](#convolution))

- [PointwiseAdd](ops.py#L37)
- [PointwiseSub](ops.py#L47)
- [PointwiseMul](ops.py#L57)
- [PointwiseDiv](ops.py#L67)
- [Sinus](ops.py#L77)
- [Cosinus](ops.py#L86)
- [Exp](ops.py#L95)
- [Relu](ops.py#L104)
- [Sigmoid](ops.py#L131)
- [Reshape](ops.py#L144)
- [Transpose](ops.py#L160)
- [Softmax](ops.py#L185)
- [SumReduce](ops.py#L221)
- [MeanReduce](ops.py#L235)
- [MaxReduce](ops.py#L250)
- [MinReduce](ops.py#L273)

## Explanations for some of the derivatives

In the calculations **L** means the loss, calculated at the end of the graph. Now, it is assumed to be a scalar.

### Gelu

This is the [original paper](https://arxiv.org/abs/1606.08415) for gelu. 
There are two formulas:
- the original one
- and approximation for better inference speed.

Here, we will only deal with the original. The formula for forward inference:

$$y = x \cdot \frac{1}{2} \left[ 1 + erf(x / \sqrt{2}) \right]$$

The error function can be defined by the following integral:

$$erf(z) = \frac{2}{\sqrt{\pi}} \int_0^z{\exp^{-t^2}dt}$$

The derivative of the error function is therefore the function inside the integral by definition:

$$erf'(t) = \frac{d erf(z)}{dz} \big|_{z=t} = \frac{2}{\sqrt{\pi}} \exp^{-t^2}$$

The derivative function for the gelu:

$$y' = \frac{1}{2}\left[ 1 + erf(x / \sqrt{2}) \right] + x \cdot \frac{1}{2} \sqrt{\frac{2}{\pi}} \exp^{-\frac{x^2}{2}}$$

### MatMul

The matrix multiplication with tensors:

$$Y = A \cdot B$$

To derive the derivative of the matrix multiplication, let's apply *tensor notation*.
The above formula is equivalent with:

$$y_{ij} = \sum_k{a_{ik} \cdot b_{kj}}$$

(Summation convention was omitted here for simplicity.)
The loss will be a function of Y, therefore the following derivative needs to be calculated with the chain rule (for A):

$$\frac{\partial L}{\partial a_{pq}} = \sum_{i,j}{\frac{\partial L}{\partial y_{ij}}\frac{\partial y_{ij}}{\partial a_{pq}}}$$

Taking into account that the derivative term is fixed, $a_{pq}$, $y_{ij}$ can be written as:

$$y_{ij} = \sum_{q}{a_{pq} \cdot b_{qj}} = y_{pj}$$

So $i=p$ for the original index. Then the derivative:

$$\frac{\partial L}{\partial a_{pq}} = \sum_{j}{g_{pj} \cdot \frac{\partial y_{pj}}{\partial a_{pq}}} = \sum_{j}{g_{pj} \cdot b_{qj}}$$

The term $g_{pj}$ is just a simplification for the derivative according to $y_{pj}$. (In the source code, g is the tg tensor.)
To create the equivalent matrix product for this tensor notation expression, we has to consider what is the effect of transposing a matrix in tensor notations:

$$B => b_{ij}: B^T => b_{ji}$$

Therefore:

$$\frac{\partial L}{\partial a_{pq}} = \sum_{j}{g_{pj} \cdot b_{qj}} => G \cdot B^T$$

As a summary for matrix A:

$$\frac{\partial L}{\partial A} = G \cdot B^T$$

A similar calculation can show the derivative according to B:

$$\frac{\partial L}{\partial B} = (G^T \cdot A)^T$$

### Convolution

The convolution is assumed to be 2-dimensional, the extension for other dimensions is straightforward.
There are three variables in convolution: x (image), weight and bias.
The derivative can be calculated by any of them.
Let's start with the bias as it is the most simplest one.

#### Bias

First, the formula for the convolution with bias (B):

$$y_{b,f,i,j} = K_{b,f,i,j} + B_f$$

The calculation of K is a convolution without bias.
We first need the derivative according to B:

$$\frac{\partial L}{\partial B_f} = \sum_{b,i,j} \frac{\partial L}{\partial y_{b,f,i,j}} \cdot \frac{\partial y_{b,f,i,j}}{\partial B_f}$$

Looking the formula for $y_{b,f,i,j}$, the second term inside the sum is 1.
Therefore:

$$\frac{\partial L}{\partial B_f} = \sum_{b,i,j} \frac{\partial L}{\partial y_{b,f,i,j}} = \sum_{b,i,j} g_{b,f,i,j} $$

Where g is just an abbreviation.
The formula means a sum reduce over 3 axes: b, i, j.
(b: batch, i: height index for images, j: width index for images)

#### Weights

The implementation is using 2d spatial dimension but here, for simpler indexing in the formulas, we will assume 1d spatial indices (1d convolution).
Assume non-trivial dilation (d) and stride (s).

The forward pass for convolution can be expressed with tensor notation as:

$$Y = Conv(X, W, B, s, d) => y_{b,f,i} = \sum_{c, r} {x_{b,c,is+rd} \cdot w_{f,c,r}} + B_f$$

We need the derivative according to the weight:

$$\frac{\partial L}{\partial w_{f, c, r}} = \sum_{b,i} {\frac{\partial L}{\partial y_{b, f, i}} \cdot \frac{\partial y_{b,f,i}}{\partial w_{f,c,r}}}$$

Now, the calculation for the second derivative:

$$\frac{\partial y_{b,f,i}}{\partial w_{f,c,r}} = x_{b,c,is+rd}$$

Introducing G for the derivative by Y, it simplifies to:

$$\frac{\partial L}{\partial w_{f, c, r}} = \sum_{b,i} {g_{b,f,i} \cdot x_{b,c,is+rd}}$$

Do some rearrangment (in indices and order), so we can compare this formula accurately to the first one (to forward conv.):

$$\frac{\partial L}{\partial w_{c, f, r}} = \sum_{b,i} {x_{c,b,rd+is} \cdot g_{f,b,i}} => Conv(X', W', B, s', d')$$

Then, the main point is to express X', W' and s', d':

$$X' = Tr(X, b \leftrightarrow c)$$
$$W' = Tr(G, b \leftrightarrow f)$$
$$s' = d$$
$$d' = s$$

The last two formula indicates, the role of the stride and dilation is interchanged.

Summary:
* the derivative of the convolution according to the kernel weights (W) is also a convolution
* this effective convolution swaps the batch and channel axis in the input tensor (image)
* the weight for the effective convolution is expressed by the gradient calculated after the convolution (or it can also be filled with ones, but the tensor shape needs to match the shape of Y)
* the original stride becomes the dilation in the effective convolution 
* the original dilation becomes the stride in the effective convolution
* a cropping can be necessary to adjust the output size of the effective convolution, the unused outputs should be cropped because the convolution can leave some of the X elements untouched when the stride and kernel size parameters are chosen a specific way

#### Image

The derivative according to the image (X) is a bit more involved. But it can be also calculated with a convolution (or with transposed convolution).

Let's start with the forward convolution in tensor notation:

$$Y = Conv(X, W, B, s, d) => y_{b,f,i} = \sum_{c, r} {x_{b,c,is+rd} \cdot w_{f,c,r}} + B_f$$

Ignore the bias and introduce a new index: $i'=is+rd$.
Then the equation becomes:

$$y_{b,f,i} = \sum_{c, r; (is+rd=i')} {x_{b,c,i'} \cdot w_{f,c,r}}$$

The derivative we are looking for:

$$\frac{\partial L}{\partial x_{b,c,i'}} = \sum_{f,i,r;(is+rd=i')} {g_{b,f,i} \cdot \frac{\partial y_{b,f,i}}{\partial x_{b,c,i'}}}$$

The second term after taking the derivative (for fix i' and i, r becomes concrete too):

$$\frac{\partial L}{\partial x_{b,c,i'}} = \sum_{f,i,r;(is+rd=i')} {g_{b,f,i} \cdot w_{f,c,r}}$$

In order to arrive a formula wich can be similar to a convolution, we have to get rid of the condition ($is+rd=i'$) and transform the formula something similar to the first equation.
Let's define a modified version of w and g:

$$w'_{f,c,r'} = \bigg\{ {\begin{matrix} w_{f,c,r}: r'=rd \\\ 0: otherwise \end{matrix}}$$

$$g'_{b,f,i''} = \bigg\{ {\begin{matrix} g_{b,f,i}: i''=is \\\ 0: otherwise \end{matrix}}$$

This is like padding the G and W matrices internally with zeros (similarly to dilation).
Then the derivative becomes:

$$\frac{\partial L}{\partial x_{b,c,i'}} = \sum_{f,i;(is+r'=i')} {g_{b,f,i} \cdot w'_{f,c,r'}} = \sum_{f,i''} {g'_{b,f,i''} \cdot w'_{f,c,i'-i''}}$$

Where $i'-i'' = rd = r'$. With some adjustment in the orders of the axes:

$$\frac{\partial L}{\partial x_{b,c,i'}} = \sum_{f,i''} {g'_{b,f,i''} \cdot w'_{c,f,i'-i''}}$$

This formula looks exactly like a convolution but due to the negative sign of $i''$ in the right term, the weights need to be rotated around the spatial axis with 180 degrees (or let's say mirror it). The G' matrix also requires external padding (with zero) to provide valid region for summation.

$$\frac{\partial L}{\partial X} = Conv(Pad(G'), Rot(Tr(W', f \leftrightarrow c)), B, 1, 1)$$

Summary:
* the derivative of the convolution in terms of X is also a convolution
* the weight of the effective convolution requires a swap of the batch and channel axis; the spatial axis needs to be reverted (or rotated when pictured in 2d)
* the G matrix requires padding with zeros among its elements with s-1 elements; (new shape will be (gh - 1)(s-1) + gh); this results in G'
* the G' matrix also requires padding outside with K - 1 zeros on both sides (K is the size of W')
* both the stride and dilation of the effective convolution are 1
* the output shape should be adjusted to the shape of X, with zero pedding on the right side

Alternative with transposed convolution:
* the steps outlined in the summary is very similar to the transposed convolution; in fact it is a transposed convolution

$$\frac{\partial L}{\partial X} = TranspConv(G, W, s, d, outpad)$$

* where outpad is the adjustment to the shape of X.

