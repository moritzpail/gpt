# GPT-2 on Shakespeare
In addition to the config that Karpathy uses in his [tutorial](https://www.youtube.com/watch?v=kCc8FmEb1nY), this notebook implements a different model hidden dimension, optimizer, and uses GeLU instead of ReLU in ffwd network of the model. In particular, we 
- Use a hidden dimension size of 128 instead of 384 to keep the training more manageable with the available resources (free GPU on Google Colab).
- Use GeLU instead of ReLU in the FFWD network of the GPT as I read that this might give improvement.
- Use the Lion optimizer as I read some [evidence](https://github.com/lucidrains/lion-pytorch) that this might also lead to efficiency gains.