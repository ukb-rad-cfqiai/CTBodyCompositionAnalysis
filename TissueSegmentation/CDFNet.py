
# Copyright 2019 Population Health Sciences and Image Analysis, German Center for Neurodegenerative Diseases(DZNE)
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

# IMPORTS
import torch
import torch.nn as nn

class CDFNet(nn.Module):
    """
    2D: Competitive Dense Fully Convolutional Network
    network implementation from @ https://arxiv.org/abs/1807.07803
    * Same Number of filters per layer (normally 64)
    * Dense Connections in blocks
    * Concatenationes are replaced with Maxout (competitive dense blocks)
    * Global skip connections are fused by Maxout (global competition) and concat operation between unpool and skip
    * Unsampling by transpose convolutions
    Updates:

    * PReLu instead of ReLu
    """
    def __init__(self, params, **kwargs):
        super(CDFNet, self).__init__()
        self.kwargs = kwargs

        # Parameters for the Descending Arm
        self.encode1 = CompetitiveEncoderBlockInput(params)
        params['num_channels'] = params['num_filters']
        self.encode2 = CompetitiveEncoderBlock(params)
        self.encode3 = CompetitiveEncoderBlock(params)
        self.encode4 = CompetitiveEncoderBlock(params)
        self.bottleneck = CompetitiveDenseBlock(params)

        # Parameters for the Ascending Arm
        params['num_channels'] = params['num_filters']
        self.decode4 = CompetitiveDecoderBlock(params)
        self.decode3 = CompetitiveDecoderBlock(params)
        self.decode2 = CompetitiveDecoderBlock(params)
        self.decode1 = CompetitiveDecoderBlock(params)

        params['num_channels'] = params['num_filters']
        self.classifier = ClassifierBlock(params)

        # Code for Network Initialization

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Computational graph
        :param tensor x: input image
        :return tensor: prediction logits
        """
        encoder_output1, skip_encoder_1 = self.encode1.forward(x)
        encoder_output2, skip_encoder_2 = self.encode2.forward(encoder_output1)
        encoder_output3, skip_encoder_3 = self.encode3.forward(encoder_output2)
        encoder_output4, skip_encoder_4 = self.encode4.forward(encoder_output3)

        bottleneck = self.bottleneck(encoder_output4)

        decoder_output4 = self.decode4.forward(bottleneck, skip_encoder_4)
        decoder_output3 = self.decode3.forward(decoder_output4, skip_encoder_3)
        decoder_output2 = self.decode2.forward(decoder_output3, skip_encoder_2)
        decoder_output1 = self.decode1.forward(decoder_output2, skip_encoder_1)

        logits = self.classifier.forward(decoder_output1)

        return logits



# Building Blocks
class CompetitiveDenseBlock(nn.Module):
    """
    Function to define a competitive dense block comprising of 3 convolutional layers, with BN/ReLU

    Inputs:
    -- Params
     params = {'num_channels': 1,
               'num_filters': 64,
               'kernel_h': 5,
               'kernel_w': 5,
               'stride_conv': 1,
               'pool': 2,
               'stride_pool': 2,
               'num_classes': 44
               'kernel_c':1
               'input':True
               }
    """

    def __init__(self, params, outblock=False):
        """
        Constructor to initialize the Competitive Dense Block
        :param dict params: dictionary with parameters specifiying block architecture
        :param bool outblock: Flag indicating if last block (before classifier block) is set up.
                               Default: False
        :return None:
        """
        super(CompetitiveDenseBlock, self).__init__()

        # Padding to get output tensor of same dimensions
        padding_h = int((params['kernel_h'] - 1) / 2)
        padding_w = int((params['kernel_w'] - 1) / 2)

        # Sub-layer output sizes for BN; and
        conv0_in_size = int(params['num_filters'])  # num_channels
        conv1_in_size = int(params['num_filters'])
        conv2_in_size = int(params['num_filters'])

        # Define the learnable layers

        self.conv0 = nn.Conv2d(in_channels=conv0_in_size, out_channels=params['num_filters'],
                               kernel_size=(params['kernel_h'], params['kernel_w']),
                               stride=params['stride_conv'], padding=(padding_h, padding_w))

        self.conv1 = nn.Conv2d(in_channels=conv1_in_size, out_channels=params['num_filters'],
                               kernel_size=(params['kernel_h'], params['kernel_w']),
                               stride=params['stride_conv'], padding=(padding_h, padding_w))

        # 1 \times 1 convolution for the last block
        self.conv2 = nn.Conv2d(in_channels=conv2_in_size, out_channels=params['num_filters'],
                               kernel_size=(1, 1),
                               stride=params['stride_conv'], padding=(0, 0))

        self.bn1 = nn.BatchNorm2d(num_features=conv1_in_size)
        self.bn2 = nn.BatchNorm2d(num_features=conv2_in_size)
        self.bn3 = nn.BatchNorm2d(num_features=conv2_in_size)

        self.prelu = nn.PReLU()  # Learnable ReLU Parameter
        self.outblock = outblock

    def forward(self, x):
        """
        CompetitiveDenseBlock's computational Graph
        {in (Conv - BN from prev. block) -> PReLU} -> {Conv -> BN -> Maxout -> PReLU} x 2 -> {Conv -> BN} -> out
        end with batch-normed output to allow maxout across skip-connections

        :param tensor x: input tensor (image or feature map)
        :return tensor out: output tensor (processed feature map)
        """
        # Activation from pooled input
        x0 = self.prelu(x)

        # Convolution block 1
        x0 = self.conv0(x0)
        x1_bn = self.bn1(x0)
        x0_bn = torch.unsqueeze(x, 4)
        x1_bn = torch.unsqueeze(x1_bn, 4)
        x1 = torch.cat((x1_bn, x0_bn), dim=4)  # Concatenate along the 5th dimension NB x C x H x W x F
        x1_max, _ = torch.max(x1, 4)
        x1 = self.prelu(x1_max)

        # Convolution block 2
        x1 = self.conv1(x1)
        x2_bn = self.bn2(x1)
        x2_bn = torch.unsqueeze(x2_bn, 4)
        x1_max = torch.unsqueeze(x1_max, 4)
        x2 = torch.cat((x2_bn, x1_max), dim=4)  # Concatenating along the 5th dimension
        x2_max, _ = torch.max(x2, 4)
        x2 = self.prelu(x2_max)

        # Convolution block 3 (end with batch-normed output to allow maxout across skip-connections)
        out = self.conv2(x2)

        if not self.outblock:
            out = self.bn3(out)

        return out


class CompetitiveDenseBlockInput(nn.Module):
    """
    Function to define a competitive dense block comprising of 3 convolutional layers, with BN/ReLU for input

    Inputs:
    -- Params
     params = {'num_channels': 1,
               'num_filters': 64,
               'kernel_h': 5,
               'kernel_w': 5,
               'stride_conv': 1,
               'pool': 2,
               'stride_pool': 2,
               'num_classes': 44
               'kernel_c':1
               'input':True
              }
    """

    def __init__(self, params):
        """
        Constructor to initialize the Competitive Dense Block
        :param dict params: dictionary with parameters specifiying block architecture
        """
        super(CompetitiveDenseBlockInput, self).__init__()

        # Padding to get output tensor of same dimensions
        padding_h = int((params['kernel_h'] - 1) / 2)
        padding_w = int((params['kernel_w'] - 1) / 2)

        # Sub-layer output sizes for BN; and
        conv0_in_size = int(params['num_channels'])
        conv1_in_size = int(params['num_filters'])
        conv2_in_size = int(params['num_filters'])

        # Define the learnable layers

        self.conv0 = nn.Conv2d(in_channels=conv0_in_size, out_channels=params['num_filters'],
                               kernel_size=(params['kernel_h'], params['kernel_w']),
                               stride=params['stride_conv'], padding=(padding_h, padding_w))

        self.conv1 = nn.Conv2d(in_channels=conv1_in_size, out_channels=params['num_filters'],
                               kernel_size=(params['kernel_h'], params['kernel_w']),
                               stride=params['stride_conv'], padding=(padding_h, padding_w))

        # 1 \times 1 convolution for the last block
        self.conv2 = nn.Conv2d(in_channels=conv2_in_size, out_channels=params['num_filters'],
                               kernel_size=(1, 1),
                               stride=params['stride_conv'], padding=(0, 0))

        self.bn0 = nn.BatchNorm2d(num_features=conv0_in_size)
        self.bn1 = nn.BatchNorm2d(num_features=conv1_in_size)
        self.bn2 = nn.BatchNorm2d(num_features=conv2_in_size)
        self.bn3 = nn.BatchNorm2d(num_features=conv2_in_size)

        self.prelu = nn.PReLU()  # Learnable ReLU Parameter

    def forward(self, x):
        """
        CompetitiveDenseBlockInput's computational Graph
        in -> BN -> {Conv -> BN -> PReLU} -> {Conv -> BN -> Maxout -> PReLU} -> {Conv -> BN} -> out

        :param tensor x: input tensor (image or feature map)
        :return tensor out: output tensor (processed feature map)
        """
        # Input batch normalization
        x0_bn = self.bn0(x)

        # Convolution block1
        x0 = self.conv0(x0_bn)
        x1_bn = self.bn1(x0)
        x1 = self.prelu(x1_bn)

        # Convolution block2
        x1 = self.conv1(x1)
        x2_bn = self.bn2(x1)
        # First Maxout
        x1_bn = torch.unsqueeze(x1_bn, 4)
        x2_bn = torch.unsqueeze(x2_bn, 4)  # Add Singleton Dimension along 5th
        x2 = torch.cat((x2_bn, x1_bn), dim=4)  # Concatenating along the 5th dimension
        x2_max, _ = torch.max(x2, 4)
        x2 = self.prelu(x2_max)

        # Convolution block 3
        out = self.conv2(x2)
        out = self.bn3(out)

        return out


class CompetitiveEncoderBlock(CompetitiveDenseBlock):
    """
    Encoder Block = CompetitiveDenseBlock + Max Pooling
    """

    def __init__(self, params):
        """
        Encoder Block initialization
        :param dict params: parameters like number of channels, stride etc.
        """
        super(CompetitiveEncoderBlock, self).__init__(params)
        self.maxpool = nn.MaxPool2d(kernel_size=params['pool'], stride=params['stride_pool'],
                                    return_indices=False)  # For Unpooling later on with the indices

    def forward(self, x):
        """
        CComputational graph for Encoder Block:
          * CompetitiveDenseBlock
          * Max Pooling (+ retain indices)

        :param tensor x: feature map from previous block
        :return: original feature map, maxpooled feature map, maxpool indices
        """
        out_block = super(CompetitiveEncoderBlock, self).forward(x)  # To be concatenated as Skip Connection
        out_encoder = self.maxpool(out_block)  # Max Pool as Input to Next Layer
        return out_encoder, out_block


class CompetitiveEncoderBlockInput(CompetitiveDenseBlockInput):
    """
    Encoder Block = CompetitiveDenseBlockInput + Max Pooling
    """

    def __init__(self, params):
        """
        Encoder Block initialization
        :param dict params: parameters like number of channels, stride etc.
        """
        super(CompetitiveEncoderBlockInput, self).__init__(params)  # The init of CompetitiveDenseBlock takes in params
        self.maxpool = nn.MaxPool2d(kernel_size=params['pool'], stride=params['stride_pool'],
                                    return_indices=False)  # For Unpooling later on with the indices

    def forward(self, x):
        """
        Computational graph for Encoder Block:
          * CompetitiveDenseBlockInput
          * Max Pooling (+ retain indices)

        :param tensor x: feature map from previous block
        :return: original feature map, maxpooled feature map, maxpool indices
        """
        out_block = super(CompetitiveEncoderBlockInput, self).forward(x)  # To be concatenated as Skip Connection
        out_encoder= self.maxpool(out_block)  # Max Pool as Input to Next Layer
        return out_encoder, out_block


class CompetitiveDecoderBlock(CompetitiveDenseBlock):
    """
    Decoder Block = (Unpooling + Skip Connection) --> Dense Block
    """

    def __init__(self, params, outblock=False):
        """
        Decoder Block initialization
        :param dict params: parameters like number of channels, stride etc.
        :param bool outblock: Flag, indicating if last block of network before classifier
                              is created. Default: False
        """
        super(CompetitiveDecoderBlock, self).__init__(params, outblock=outblock)

        # Padding to get output tensor of same dimensions


        padding_input_h = int((params['kernel_h']) / 2)
        padding_input_w = int((params['kernel_w']) / 2)

        padding_output_h = int(params['kernel_h'] % 2)
        padding_output_w = int(params['kernel_w'] % 2)

        self.convT = nn.ConvTranspose2d(in_channels=params['num_filters'], out_channels=params['num_filters'],
                                        stride=params['stride_pool'],
                                        kernel_size=(params['kernel_h'], params['kernel_w']),
                                        padding=(padding_input_h, padding_input_w),
                                        output_padding=(padding_output_h, padding_output_w))

        self.conv = nn.Conv2d(in_channels=2*params['num_filters'], out_channels=params['num_filters'],
                               kernel_size=(1, 1),
                               stride=params['stride_conv'])

        self.bn=nn.BatchNorm2d(num_features=params['num_filters'])
        self.prelu = nn.PReLU()  # Learnable ReLU Parameter

    def forward(self, x, out_block):
        """
        Computational graph Decoder block:
          * Unpooling of feature maps from lower block
          * Maxout combination of unpooled map + skip connection
          * Forwarding toward CompetitiveDenseBlock

        :param tensor x: input feature map from lower block (gets unpooled and maxed with out_block)
        :param tensor out_block: skip connection feature map from the corresponding Encoder
        :param tensor indices: indices for unpooling from the corresponding Encoder (maxpool op)
        :return: processed feature maps
        """

        convT = self.convT(x)
        #print(convT.shape)
        #Merge conntections + conv + bn
        merged_unpool=torch.cat((convT,out_block),dim=1)

        x1=self.prelu(merged_unpool)
        x1=self.conv(x1)
        x1_bn=self.bn(x1)

        #maxout between skip and merge unpool
        out_block = torch.unsqueeze(out_block, 4)
        x1_bn = torch.unsqueeze(x1_bn, 4)

        concat = torch.cat((x1_bn, out_block), dim=4)  # Competitive Concatenation
        concat_max, _ = torch.max(concat, 4)
        out_block = super(CompetitiveDecoderBlock, self).forward(concat_max)

        return out_block


class ClassifierBlock(nn.Module):
    """
    Classification Block
    """
    def __init__(self, params):
        """
        Classifier Block initialization
        :param dict params: parameters like number of channels, stride etc.
        """
        super(ClassifierBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=params['num_channels'], out_channels=params['num_classes'],kernel_size= params['kernel_c'],
                              stride=params['stride_conv'])  # To generate logits

    def forward(self, x):
        """
        Computational graph of classifier
        :param tensor x: output of last CompetitiveDenseDecoder Block-
        :return: logits
        """
        logits = self.conv(x)

        return logits
