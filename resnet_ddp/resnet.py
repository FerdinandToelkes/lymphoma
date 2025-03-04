import torch
import torch.nn as nn

from torch.optim import lr_scheduler

# Sources: 
# https://blog.paperspace.com/writing-resnet-from-scratch-in-pytorch/ ,
# https://github.com/pytorch/vision/blob/1aef87d01eec2c0989458387fa04baebcc86ea7b/torchvision/models/resnet.py#L75


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        """
        in_channels: input channels
        out_channels: output channels
        stride: stride of the first convolutional layer. Default: 1 (determines how many units the filter shifts at each step)
        downsample: downsample function. Default: None
        """
        super(BasicBlock, self).__init__() # calls the __init__ method of the parent class 
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1) # padding=1 -> input size = output size
        self.bn1 = nn.BatchNorm2d(out_channels) 
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True) # inplace=True -> input is modified directly, no additional memory is used
        self.out_channels = out_channels
        
    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # downsample needed if the number of #input != #output channels or if the stride != 1
        if self.downsample:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
    
class BasicBlock1d(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        """
        The 1D version of the BasicBlock class.

        in_channels: input channels
        out_channels: output channels
        stride: stride of the first convolutional layer. Default: 1 (determines how many units the filter shifts at each step)
        downsample: downsample function. Default: None
        """
        super(BasicBlock1d, self).__init__() # calls the __init__ method of the parent class 
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1) # padding=1 -> input size = output size
        self.bn1 = nn.BatchNorm1d(out_channels) 
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True) # inplace=True -> input is modified directly, no additional memory is used
        self.out_channels = out_channels
        
    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # downsample needed if the number of #input != #output channels or if the stride != 1
        if self.downsample:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
    

class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        width = out_channels // 4
        self.conv1 = nn.Conv2d(in_channels, width, kernel_size=1, stride=stride, padding=0)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True) # inplace=True -> input is modified directly, no additional memory is used
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
    
class Bottleneck1d(nn.Module):
    """ The 1D version of the Bottleneck class. """
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck1d, self).__init__()
        width = out_channels // 4
        self.conv1 = nn.Conv1d(in_channels, width, kernel_size=1, stride=stride, padding=0)
        self.bn1 = nn.BatchNorm1d(width)
        self.conv2 = nn.Conv1d(width, width, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(width)
        self.conv3 = nn.Conv1d(width, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True) # inplace=True -> input is modified directly, no additional memory is used
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
    

class ResNet(nn.Module):
    def __init__(self, block, layers, input_channels=3, num_classes=7): # unknown class is also included
        super(ResNet, self).__init__() 
        self.inplanes = 64 # number of input channels after the first convolutional layer
        self.conv1 = nn.Conv2d(input_channels, self.inplanes, kernel_size=7, stride=2, padding=3) 
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) # pooling to reduce the spatial dimensions of the output volume 
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2) # stride=2 -> spatial dimensions are reduced by half
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) # -> invarient to input size, output size is 1x1x512
        self.fc = nn.Linear(512, num_classes)
        
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            # 1x1 convolution for downsampling
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride), 
                nn.BatchNorm2d(planes),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample)) # only the first block has downsample != None
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1) # flatten everything except batch dimension
        x = self.fc(x)
        return x
    

class ResNet1d(nn.Module):
    """Constructs a ResNet model for 1D input data such as the embeddings of the patches."""
    def __init__(self, block, layers, input_channels=1, num_classes=7): # unknown class is also included
        super(ResNet1d, self).__init__() 
        self.inplanes = 64 # number of input channels after the first convolutional layer
        self.conv1 = nn.Conv1d(input_channels, self.inplanes, kernel_size=7, stride=2, padding=3) 
        self.bn1 = nn.BatchNorm1d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1) # pooling to reduce the spatial dimensions of the output volume 
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2) # stride=2 -> spatial dimensions are reduced by half
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1) # -> invarient to input size, output size is 1x1x512
        self.fc = nn.Linear(512, num_classes)
        
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            # 1x1 convolution for downsampling
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes, kernel_size=1, stride=stride), 
                nn.BatchNorm1d(planes),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample)) # only the first block has downsample != None
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    
    def forward(self, x):
        # Add a channel dimension to the input tensor needed for the convolutional layer
        x = x.unsqueeze(1)  # New shape will be [batch_size, 1, length]

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x) 
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1) # flatten everything except batch dimension
        x = self.fc(x)
        return x
    

class ResNet1dMultipleChannels(nn.Module):
    """Constructs a ResNet model for 1D input data such as the embeddings of the patches."""
    def __init__(self, block, layers, input_channels, num_classes=7): # unknown class is also included
        super(ResNet1dMultipleChannels, self).__init__() 
        self.inplanes = 64 # number of input channels after the first convolutional layer
        self.conv1 = nn.Conv1d(input_channels, self.inplanes, kernel_size=7, stride=2, padding=3) 
        self.bn1 = nn.BatchNorm1d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1) # pooling to reduce the spatial dimensions of the output volume 
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2) # stride=2 -> spatial dimensions are reduced by half
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1) # -> invarient to input size, output size is 1x1x512
        self.fc = nn.Linear(512, num_classes)
        
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            # 1x1 convolution for downsampling
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes, kernel_size=1, stride=stride), 
                nn.BatchNorm1d(planes),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample)) # only the first block has downsample != None
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    
    def forward(self, x):
        # switch the channel dimension to the front
        x = x.permute(0, 2, 1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x) 
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1) # flatten everything except batch dimension
        x = self.fc(x)
        return x
    

def resnet18(**kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    """
    # extract arguments from kwargs, defaulting to '2d' and 3 if it is not provided
    resnet_dim = kwargs.pop('resnet_dim', '2d')
    input_channels = kwargs.pop('input_channels', 3)
    if resnet_dim == '2d':
        return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    elif resnet_dim == '1d':
        return ResNet1d(BasicBlock1d, [2, 2, 2, 2], **kwargs)
    elif resnet_dim == '1d_multiple_channels':
        return ResNet1dMultipleChannels(BasicBlock1d, [2, 2, 2, 2], input_channels, **kwargs)
    else:
        raise ValueError("Invalid resnet type. Possible values: '2d', '1d', '1d_multiple_channels'.")


def resnet34(**kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    """
    resnet_dim = kwargs.pop('resnet_dim', '2d')
    input_channels = kwargs.pop('input_channels', 3)
    if resnet_dim == '2d':
        return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    elif resnet_dim == '1d':
        return ResNet1d(BasicBlock1d, [3, 4, 6, 3], **kwargs)
    elif resnet_dim == '1d_multiple_channels':
        return ResNet1dMultipleChannels(BasicBlock1d, [3, 4, 6, 3], input_channels, **kwargs)
    else:
        raise ValueError("Invalid resnet type. Possible values: '2d', '1d', '1d_multiple_channels'.")


def resnet50(**kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    """
    resnet_dim = kwargs.pop('resnet_dim', '2d')
    input_channels = kwargs.pop('input_channels', 3)
    if resnet_dim == '2d':
        return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    elif resnet_dim == '1d':
        return ResNet1d(Bottleneck1d, [3, 4, 6, 3], **kwargs)
    elif resnet_dim == '1d_multiple_channels':
        return ResNet1dMultipleChannels(Bottleneck1d, [3, 4, 6, 3], input_channels, **kwargs)
    else:
        raise ValueError("Invalid resnet type. Possible values: '2d', '1d', '1d_multiple_channels'.")


def resnet101(**kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    """
    resnet_dim = kwargs.pop('resnet_dim', '2d')
    input_channels = kwargs.pop('input_channels', 3)
    if resnet_dim == '2d':
        return ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    elif resnet_dim == '1d':
        return ResNet1d(Bottleneck1d, [3, 4, 23, 3], **kwargs)
    elif resnet_dim == '1d_multiple_channels':
        return ResNet1dMultipleChannels(Bottleneck1d, [3, 4, 23, 3], input_channels, **kwargs)
    else:
        raise ValueError("Invalid resnet type. Possible values: '2d', '1d', '1d_multiple_channels'.")


def resnet152(**kwargs):
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    """
    resnet_dim = kwargs.pop('resnet_dim', '2d')
    input_channels = kwargs.pop('input_channels', 3)
    if resnet_dim == '2d':
        return ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    elif resnet_dim == '1d':
        return ResNet1d(Bottleneck1d, [3, 8, 36, 3], **kwargs)
    elif resnet_dim == '1d_multiple_channels':
        return ResNet1dMultipleChannels(Bottleneck1d, [3, 8, 36, 3], input_channels, **kwargs)
    else:
        raise ValueError("Invalid resnet type. Possible values: '2d', '1d', '1d_multiple_channels'.")

    
############################################################################################################################################
############################################### FUNCTIONS FOR LOADING THE MODEL ############################################################
############################################################################################################################################    



def load_train_objs(resnet_type: str, resnet_dim: str, input_channels: int, gpu_id: int,
                    learning_rate: float, momentum: float = 0.9, step_size: int = 7, gamma: float = 0.1,
                    warmup_epochs: int = 0):
    """
    Loads the model, optimizer and scheduler for training.

    Args:
        resnet_type (str): Type of resnet to use. Possible values: "resnet18", "resnet34", "resnet50", "resnet101", "resnet152".
        resnet_dim (str): Dimension of the resnet. Possible values: "2d", "1d", "1d_multiple_channels".
        input_channels (int): Number of input channels.
        gpu_id (int): ID of the GPU to use.
        learning_rate (float): Learning rate for the optimizer.
        momentum (float, optional): Momentum for the optimizer. Default: 0.9.
        step_size (int, optional): Step size for the scheduler. Default: 7.
        gamma (float, optional): Decay factor for the scheduler. Default: 0.1.
        warmup_epochs (int, optional): Number of warm-up epochs. Default: 0.

    Returns:
        tuple: Tuple containing the model, optimizer and scheduler.
    """
    # select and load the model
    if resnet_type == "resnet18":
        model = resnet18(resnet_dim=resnet_dim, input_channels=input_channels)
    elif resnet_type == "resnet34":
        model = resnet34(resnet_dim=resnet_dim, input_channels=input_channels)
    elif resnet_type == "resnet50":
        model = resnet50(resnet_dim=resnet_dim, input_channels=input_channels)
    elif resnet_type == "resnet101":
        model = resnet101(resnet_dim=resnet_dim, input_channels=input_channels)
    elif resnet_type == "resnet152":
        model = resnet152(resnet_dim=resnet_dim, input_channels=input_channels)
    else:
        raise ValueError("Invalid resnet type. Possible values: 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'.")
    # setup optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    # Define warm-up LR scheduler
    def warmup_function(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs            
        return 1  # After warm-up, use the base learning rate

    warmup_scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_function)

    # Main LR scheduler (StepLR)
    main_scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    # move model to GPU and convert BatchNorm to SyncBatchNorm (see https://pytorch.org/tutorials/beginner/ddp_series_multigpu.html)
    model = model.to(gpu_id)
    torch.nn.SyncBatchNorm.convert_sync_batchnorm(model) 
    return model, optimizer, warmup_scheduler, main_scheduler

def load_trained_model(resnet_type: str, resnet_dim: str, input_channels: int, path_to_model: str, gpu_id: int):
    """
    Loads the trained model.

    Args:
        resnet_type (str): Type of resnet to use. Possible values: "resnet18", "resnet34", "resnet50", "resnet101", "resnet152".
        resnet_dim (str): Dimension of the resnet. Possible values: "2d", "1d", "1d_multiple_channels".
        input_channels (int): Number of input channels.
        path_to_model (str): Path to the trained model.
        gpu_id (int): ID of the GPU to use.

    Returns:
        nn.Module: The trained model.
    """
    # select and load the model
    if resnet_type == "resnet18":
        model = resnet18(resnet_dim=resnet_dim, input_channels=input_channels)
    elif resnet_type == "resnet34":
        model = resnet34(resnet_dim=resnet_dim, input_channels=input_channels)
    elif resnet_type == "resnet50":
        model = resnet50(resnet_dim=resnet_dim, input_channels=input_channels)
    elif resnet_type == "resnet101":
        model = resnet101(resnet_dim=resnet_dim, input_channels=input_channels)
    elif resnet_type == "resnet152":
        model = resnet152(resnet_dim=resnet_dim, input_channels=input_channels)
    else:
        raise ValueError("Invalid resnet type. Possible values: 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'.")
    # load the trained model
    trained_model = torch.load(path_to_model)
    model.load_state_dict(trained_model["MODEL_STATE"])
    # move model to GPU and convert BatchNorm to SyncBatchNorm (see https://pytorch.org/tutorials/beginner/ddp_series_multigpu.html)
    model = model.to(gpu_id)
    torch.nn.SyncBatchNorm.convert_sync_batchnorm(model) 
    return model


class Connector(nn.Module):
    """ Mini model to connect the output of the ResNet model to the final output. """
    def __init__(self, num_classes=7): # unknown class is also included
        super().__init__()
        self.inplanes = 64 # number of input channels after the first convolutional layer
        self.conv1 = nn.Conv1d(1, self.inplanes, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(self.inplanes)
        self.conv2 = nn.Conv1d(self.inplanes, self.inplanes, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1) # pooling to reduce the spatial dimensions of the output volume 
        self.avgpool = nn.AdaptiveAvgPool1d(1) # -> invarient to input size, output size is 1x1x512
        self.fc = nn.Linear(self.inplanes, num_classes)
            
    
    def forward(self, x):
        # Add a channel dimension to the input tensor needed for the convolutional layer
        x = x.unsqueeze(1)  # New shape will be [batch_size, 1, length]
        # first block where dimensions are reduced by half twice (stride=2 in the conv1 and maxpool layer)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x) 
        x = self.maxpool(x)
        # second block where dimensions stay the same
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.avgpool(x) # reduce spatial dimensions to (bs, #filters, 1, 1)
        x = torch.flatten(x, start_dim=1) # flatten everything except batch dimension

        x = self.fc(x)
        return x

if __name__ == "__main__":
    # Define model
    model = resnet18()

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=0.001, momentum=0.9)  

    #print(model)
    print("Number of parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model, optimizer, and learning rate schedulers
    warmup_epochs = 5  # Should match the warmup_epochs in load_train_objs()
    model, optimizer, warmup_scheduler, main_scheduler = load_train_objs("resnet18", "1d", 1, device, 0.01, warmup_epochs=warmup_epochs)

    print("Number of parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

    # Define training loop
    num_epochs = 15  # Example: Run for 15 epochs

    for epoch in range(num_epochs):
        # Print learning rate before the epoch
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}: Learning rate before step = {current_lr:.6f}")

        # Dummy training step
        optimizer.step()  # Normally, one would compute loss.backward() before this

        # Apply warm-up or main scheduler
        if epoch < warmup_epochs:
            warmup_scheduler.step()
        else:
            main_scheduler.step()

        # Print learning rate after the scheduler step
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}: Learning rate after step = {current_lr:.6f}\n")


  

