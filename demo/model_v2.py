import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint_adjoint as odeint

# ------
class MultimodalNetwork(nn.Module):
    def __init__(self, tabular_data_size, n_classes=3):
        super(MultimodalNetwork, self).__init__()
        self.u = nn.Parameter(torch.ones(4))
        # Tabular data branch
        self.tabular_branch = nn.Sequential(
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
        self.tabular_classifier = nn.Linear(16, n_classes)
        
        # Genetic data branch, treating as flat input for simplicity
        self.genetic_branch = nn.Sequential(
            nn.Linear(500*6, 1024),
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
        self.genetic_classifier = nn.Linear(64, n_classes)
        self.t_g_layer = nn.Linear(16 + 64, 32)
        
        # Image data branch (3D CNN for simplicity, adjust as needed)
        self.image_branch = nn.Sequential(
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
        self.image_classifier = nn.Linear(32 * 128 * 128, n_classes)  # Adjust size accordingly
        
        # Classification module
        self.classifier = nn.Sequential(nn.Linear(32 + 32 * 128 * 128, 1024),
                                        nn.ReLU(),
                                        nn.Linear(1024, 128),
                                        nn.LeakyReLU(),
                                        nn.Linear(128, n_classes),
                                        )
        
        # self.criterion = nn.CrossEntropyLoss()
        self.weight_tab = nn.Parameter(torch.randn(3))
        #self.weight_gen = nn.Parameter(torch.randn(3))
        #self.weight_img = nn.Parameter(torch.randn(3))

    def forward(self, tabular_data, genetic_data, image_data, labels):
        # print(tabular_data.shape)
        # print(genetic_data.shape)
        # print(image_data.shape)
        
        tabular_out = self.tabular_branch(tabular_data)
        genetic_out = self.genetic_branch(genetic_data.view(-1, 500*6))
        image_out = self.image_branch(image_data)
        
        tabular_cls = self.tabular_classifier(tabular_out)
        genetic_cls = self.genetic_classifier(genetic_out)
        #print(image_out.shape)
        image_cls = self.image_classifier(image_out)
        
        tabular_cls = self.weight_tab[0] * tabular_cls
        genetic_cls = self.weight_tab[1] * genetic_cls
        image_cls = self.weight_tab[2] * image_cls
	
        t_g_fused = torch.cat((tabular_out, genetic_out), dim=1)
        t_g_out = self.t_g_layer(t_g_fused)
        fused_representation = torch.cat((t_g_out, image_out), dim=1)

        #print("fused_representation shape:", fused_representation.shape)
        
        # Classification
        output = self.classifier(fused_representation)
        #print("probs:",output)
        # print(tabular_cls.shape, labels.shape)
        # Calculate individual losses for each branch
        #loss_t = self.criterion(tabular_cls, labels)
        #loss_g = self.criterion(genetic_cls, labels)
        #loss_i = self.criterion(image_cls, labels)
        
        # Calculate the total loss
        #loss = self.criterion(output, labels)
        #weights = torch.softmax(self.u, dim=0)
        #total_loss = weights[0] * loss_t + weights[1] * loss_g + weights[2] * loss_i + weights[3] * loss

        return tabular_cls, genetic_cls, image_cls, output

'''
print("==== Passing test case in Model =======")
# Example instantiation and forward pass
tabular_data_size = 65
n_classes = 3

model = MultimodalNetwork(tabular_data_size, n_classes)
# print(model)

# Example data
tabular_data = torch.randn(1, tabular_data_size)
genetic_data = torch.randn(1, 500, 6)
image_data = torch.randn(1, 1, 64, 128, 128)
labels = torch.tensor([1])

# Forward pass
tabular_cls, genetic_cls, image_cls, output = model(tabular_data, genetic_data, image_data, labels)
#print("Weights:", weights)
# print("Total Loss:", total_loss)
print("Output:", output)

print("================= MODEL OK! ============ ")
'''