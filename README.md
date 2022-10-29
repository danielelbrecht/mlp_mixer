# mlp_mixer
Implementation of MLP Mixer in Pytorch

# About
MLP Mixer is an all MLP mixer architecture for computer vision, developed by the
researchers in this publication: https://arxiv.org/abs/2105.01601

The core idea is to build a powerful computer vision classifier
using a simple MLP-based architecture.  Similar to vision transformers,
The first stage of the model is a patch embedding layer.  This is simply a convolutional 
layer where the stride is equal to the patch size, so the image is processed patch-wise
.  The output of this layer is a grid of patch representations where the number of channels 
in the convolutional layer represents the representation dimension, or embedding dimension.
This grid can be flattened into a sequence of patches, making a matrix with dimension [embedding dim, # patches]

```
patch_embedding = nn.Conv2d(img_shape[0], embedding_dim, kernel_size=patch_size, stride=patch_size)
```

The core of the model is the Mixer module, which takes in this sequence of patches
as input.  The Mixer module has two MLP sub-modules, the token mixer and channel mixer.
The token mixer is applied along the patch dimension, and learns (or 'mixes') information 
between patches/tokens.  The channel mixer does the same across the embedding dimension.  Residual 
connections and layer normalization increase the performance of the module.
These modules can be stacked sequentially to create a deep network.  The modules are
isotropic, meaning that the input to the mixer module has the same shape as the output.


```
class Mixer(nn.Module):

    def __init__(self, patch_dim, embedding_dim, mlp_dim):
        super().__init__()
        self.ln1 = nn.LayerNorm(patch_dim)
        self.mlp1 = MLP(embedding_dim, mlp_dim)

        self.ln2 = nn.LayerNorm(patch_dim)
        self.mlp2 = MLP(patch_dim, mlp_dim)

    def __call__(self, x):
        """
        Input shape: [Batch size, embedding dim, patch dim]
        """

        out = self.ln1(x)
        out = out.permute(0, 2, 1)  # Output shape: [Batch size, patch dim, embedding dim]
        out = self.mlp1(out)
        out = out.permute(0, 2, 1)  # Output shape: [Batch size, embedding dim, patch dim]

        summed = out + x

        out = self.ln2(summed)
        out = self.mlp2(out)

        return out + summed


class MLP(nn.Module):

    def __init__(self, input_dim, mlp_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, mlp_dim)  # First fully connected
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(mlp_dim, input_dim)

    def __call__(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        return x
```


Finally, there is a simple, multi-layer perceptron classification module at the end, which takes
in the flattened patches/channels matrix.

```
clf = nn.Sequential(nn.Linear(in_features=patch_dim*embedding_dim, out_features=clf_dim),
                    nn.GELU(),
                    nn.Linear(clf_dim, clf_dim),
                    nn.GELU(),
                    nn.Linear(clf_dim, n_classes))
```

# Usage
mlp_mixer.py - contains the model implementation

main.py - applies MLP mixer to MNIST dataset


A MLPMixer model can be used in the following way:
```
from mlp_mixer import MLPMixer

model = MLPMixer(img_shape=(1, 28, 28),
                 patch_size=7,
                 embedding_dim=128,
                 n_layers=3,
                 mlp_dim=64,
                 clf_dim=512,
                 n_classes=10)
                 
```
The initialization parameters are:

image_shape: The shape of an input image, with channels first.
For RGB images the first dimension would be 3, but since MNIST is grayscale
we use 1.

patch_size: Size of the patch for the patch embedding

embedding dim: Number of channels in the convolutional layer
for patch embedding

n_layers: Number of Mixer modules to use in model

mlp_dim: MLP dimension for the MLP modules within Mixer

clf_dim: Number of units for the linear layers in the classifier module

n_classes: Number of classes in dataset, 10 for MNIST

# Results
In main.py train a MLPMixer model on the MNIST dataset, which achieves
97% accuracy after the first training epoch, and converges
around 99% tet accuracy. THe model achieves very competitive results on ImageNet,
but I do not have the compute for that.