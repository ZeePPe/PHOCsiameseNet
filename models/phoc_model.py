import torch
import torch.nn as nn
import torch.nn.functional as F
from models.myphocnet import PHOCNet
from models.spatial_pyramid_layer import GPP


class PHOC(nn.Module):
    '''
    Network class for generating SPHOC architecture
    '''

    def __init__(self, enocder_type='phoc', n_out=1980, gpp_type='spp', max_len=10, pooling_levels=3, input_channel=1):
        super(PHOC, self).__init__()
        # some sanity checks
        if gpp_type not in ['spp', 'tpp', 'gpp']:
            raise ValueError('Unknown pooling_type. Must be either \'gpp\', \'spp\' or \'tpp\'')

        batchNorm_momentum = 0.1
        self.len = max_len
        #self.decoder = decoder
        #self.length_embedding = False
        #self.pos_embedding = position_embedding
        self.retval ={}

        if enocder_type=='phoc':
            self.features = PHOCNet(input_channels=input_channel)
            self.nChannels = 512
            self.pos_len = 26
            self.H =8
            self.W =18
        
        # create the spatial pooling layer
        self.pooling_layer_fn = GPP(gpp_type=gpp_type, levels=pooling_levels, pool_type='max_pool')

        pooling_output_size = self.pooling_layer_fn.pooling_output_size
        self.fc5 = nn.Linear(pooling_output_size, 4096)
        self.fc6 = nn.Linear(4096, 4096)
        self.fc7 = nn.Linear(4096, n_out)

    



    def length_embed(self, filters, length_vec):

        batch_size, _, ht, wd = filters.shape
        # print(filters.shape)
        length_vec = length_vec.unsqueeze(-1)
        # print('length_emb')
        # print(length_vec.shape)
        length_vec = length_vec.expand(-1, -1, ht * wd)
        # print(length_vec.shape)
        length_vec = length_vec.view(batch_size, -1, ht, wd)
        # print(length_vec.shape)
        # length_vec = length_vec.unsqueeze(0).expand(batch_size, -1, -1, -1)
        filters = torch.cat((filters, length_vec), dim=1)
        return filters

    def position_embed(self, y_filters):
        batch_size, _, ht, wd = y_filters.shape
        pos_vec = torch.zeros(ht, wd, ht+wd).float().cuda()
        for i in range(ht):
            for j in range(wd):
                pos_vec[i, j, i] = 1
                pos_vec[i, j, ht+j] = 1
        pos_vec = pos_vec.permute(2, 0, 1)
        pos_vec = pos_vec.unsqueeze(0).expand(batch_size, -1, -1, -1)

        filters = torch.cat((y_filters, pos_vec), dim=1)
        # filters = torch.cat((filters, length_vec), dim=1)
        return filters

    def forward(self, x):
        retval = {}
        conv_filters = self.features(x)

        y = self.pooling_layer_fn.forward(conv_filters)
        y = F.relu(self.fc5(y))
        y = F.dropout(y, p=0.5, training=self.training)
        y = F.relu(self.fc6(y))
        y = F.dropout(y, p=0.5, training=self.training)
        y = self.fc7(y)
        
        retval['phoc'] = y

        return retval

    def init_weights(self):
        self.apply(PHOC._init_weights_he)

    '''
    @staticmethod
    def _init_weights_he(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            #nn.init.kaiming_normal(m.weight.data)
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, (2. / n)**(1/2.0))
            if hasattr(m, 'bias'):
                nn.init.constant(m.bias.data, 0)
    '''

    @staticmethod
    def _init_weights_he(m):
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, (2. / n) ** (1 / 2.0))
        if isinstance(m, nn.Linear):
            n = m.out_features
            m.weight.data.normal_(0, (2. / n) ** (1 / 2.0))
            # nn.init.kaiming_normal(m.weight.data)
            nn.init.constant_(m.bias.data, 0)
