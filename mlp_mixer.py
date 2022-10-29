import torch
import torch.nn as nn


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


class MLPMixer(nn.Module):

    def __init__(self, img_shape, patch_size, embedding_dim, n_layers, mlp_dim, clf_dim, n_classes):

        """

        :param patch_size: Patch
        :param embedding_dim: Number of channels i n
        :param n_layers:
        :param mlp_dim:
        :param clf_dim:
        :param n_classes:
        """
        super().__init__()

        patch_dim = (img_shape[1] // patch_size) * (img_shape[2] // patch_size)
        self.patch_embedding = nn.Conv2d(img_shape[0], embedding_dim, kernel_size=patch_size, stride=patch_size)

        self.mixer_layers = nn.Sequential()
        for i in range(n_layers):
            mixer_layer = Mixer(patch_dim=patch_dim, embedding_dim=embedding_dim, mlp_dim=mlp_dim)
            self.mixer_layers.append(mixer_layer)

        self.clf = nn.Sequential(nn.Linear(in_features=patch_dim*embedding_dim, out_features=clf_dim),
                    nn.GELU(),
                    nn.Linear(clf_dim, clf_dim),
                    nn.GELU(),
                    nn.Linear(clf_dim, n_classes))

    def __call__(self, x):
        x = self.patch_embedding(x)  # Patch embedding

        x = x.view(x.size(0), x.size(1), -1)  # Combine H and W dimensions to get sequence of patches

        x = self.mixer_layers(x)

        x = x.view(x.size(0), -1)  # Flatten: combine embedding and patch dim

        x = self.clf(x)

        return x

