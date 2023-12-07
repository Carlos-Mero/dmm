import matplotlib.pyplot as plt
import torch

def show_cnn(model):
    conv_weights = None
    conv_layers = set()  
    for name, param in model.named_parameters():
        if 'conv' in name and 'weight' in name:
            conv_layer_name = name.split('.')[0]  
            if conv_layer_name not in conv_layers:
                conv_weights = param.data
                num_channels = conv_weights.shape[0] 
                num_cols = 4 
                num_rows = (num_channels + num_cols - 1) // num_cols

                fig, axs = plt.subplots(num_rows, num_cols, figsize=(12, 8))
                fig.suptitle(f'{name} Visualization')
                
                for i in range(num_channels):
                    row = i // num_cols
                    col = i % num_cols
                    axs[row, col].imshow(conv_weights[i,0].cpu().detach().numpy(), cmap='bwr')
                    axs[row, col].set_title(f'Channel {i}')

                plt.tight_layout()
                plt.show()
                
                conv_layers.add(conv_layer_name)

def show_bias(model):
    bias = None
    for name, param in model.named_parameters():
        if 'bias' in name:
            bias = param.data
            plt.figure(figsize=(8, 6))
            plt.title(f'{name} Visualization')
            sns.barplot(x=list(range(len(bias))), y=bias.cpu().detach().numpy())
            plt.xlabel('Channel')
            plt.ylabel('Bias Value')
            plt.show()