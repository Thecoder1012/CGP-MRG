import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint_adjoint as odeint

class MultimodalNetwork(nn.Module):
    def __init__(self, tabular_data_size, n_classes=3):
        """
        Initializes the multimodal network which integrates tabular, genetic, and image data.
        
        Args:
            tabular_data_size (int): Number of features in the tabular data.
            n_classes (int): Number of classes for classification.
        """
        super(MultimodalNetwork, self).__init__()
        self.u = nn.Parameter(torch.ones(4))  # Weight parameter for loss combination

        # Tabular data branch
        self.tabular_branch = self._build_tabular_branch(tabular_data_size)
        self.tabular_classifier = nn.Linear(16, n_classes)
        
        # Genetic data branch
        self.genetic_branch = self._build_genetic_branch()
        self.genetic_classifier = nn.Linear(64, n_classes)
        
        # Intermediate layer for fusion of tabular and genetic data
        self.t_g_layer = nn.Linear(16 + 64, 32)
        
        # Image data branch
        self.image_branch = self._build_image_branch()
        self.image_classifier = nn.Linear(32 * 128 * 128, n_classes)  # Adjust size accordingly
        
        # Final classification module
        self.classifier = nn.Sequential(
            nn.Linear(32 + 32 * 128 * 128, 1024),
            nn.ReLU(),
            nn.Linear(1024, 128),
            nn.LeakyReLU(),
            nn.Linear(128, n_classes),
        )
        
        self.weight_tab = nn.Parameter(torch.randn(3))  # Weights for tabular, genetic, and image branches

    def _build_tabular_branch(self, tabular_data_size):
        """Builds the sequential model for the tabular data branch."""
        return nn.Sequential(
            nn.Linear(tabular_data_size, 32),
            nn.InstanceNorm1d(32),
            nn.LeakyReLU(),
            nn.Linear(32, 32),
            nn.InstanceNorm1d(32),
            nn.PReLU(),
            nn.Linear(32, 32),
            nn.InstanceNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.InstanceNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.InstanceNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.InstanceNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.InstanceNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.InstanceNorm1d(16),
            nn.ReLU()
        )

    def _build_genetic_branch(self):
        """Builds the sequential model for the genetic data branch."""
        return nn.Sequential(
            nn.Linear(500 * 6, 1024),
            nn.InstanceNorm1d(1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 512),
            nn.InstanceNorm1d(512),
            nn.PReLU(),
            nn.Linear(512, 512),
            nn.InstanceNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.InstanceNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.InstanceNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.InstanceNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.InstanceNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.InstanceNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.InstanceNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.InstanceNorm1d(64),
            nn.ReLU()
        )

    def _build_image_branch(self):
        """Builds the sequential model for the image data branch."""
        return nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(16),
            nn.LeakyReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(),
            nn.Flatten()
        )

    def forward(self, tabular_data, genetic_data, image_data, labels):
        """
        Forward pass through the multimodal network.
        
        Args:
            tabular_data (torch.Tensor): Input tabular data.
            genetic_data (torch.Tensor): Input genetic data.
            image_data (torch.Tensor): Input image data.
            labels (torch.Tensor): Ground truth labels.
        
        Returns:
            tuple: Output predictions from tabular, genetic, image branches, and the final combined output.
        """
        tabular_out = self.tabular_branch(tabular_data)
        genetic_out = self.genetic_branch(genetic_data.view(-1, 500 * 6))
        image_out = self.image_branch(image_data)
        
        tabular_cls = self.weight_tab[0] * self.tabular_classifier(tabular_out)
        genetic_cls = self.weight_tab[1] * self.genetic_classifier(genetic_out)
        image_cls = self.weight_tab[2] * self.image_classifier(image_out)
        
        # Fuse tabular and genetic outputs, then combine with image features
        t_g_fused = torch.cat((tabular_out, genetic_out), dim=1)
        t_g_out = self.t_g_layer(t_g_fused)
        fused_representation = torch.cat((t_g_out, image_out), dim=1)
        
        # Final classification
        output = self.classifier(fused_representation)

        return tabular_cls, genetic_cls, image_cls, output

# Example usage
'''
print("==== Passing test case in Model =======")

# Example instantiation and forward pass
tabular_data_size = 65
n_classes = 3
model = MultimodalNetwork(tabular_data_size, n_classes)

# Example data
tabular_data = torch.randn(1, tabular_data_size)
genetic_data = torch.randn(1, 500, 6)
image_data = torch.randn(1, 1, 64, 128, 128)
labels = torch.tensor([1])

# Forward pass
tabular_cls, genetic_cls, image_cls, output = model(tabular_data, genetic_data, image_data, labels)
print("Output:", output)
print("================= MODEL OK! ============ ")
'''
