import torch
import torch.nn as nn
import torch.autograd as autograd
import os
import pickle as pkl


class GAN(nn.Module):

    def __init__(self,
                 model: nn.Module,
                 warm_features: list,
                 train_loader,
                 device,
                 item_id_name='item_id',
                 emb_dim=16,
                 c_feature='x_fre',
                 c_class_size=19):
        super(GAN, self).__init__()
        self.build(model, warm_features, train_loader, device, item_id_name, emb_dim, c_class_size)
        self.c_feature = c_feature
        return

    def build(self,
              model: nn.Module,
              item_features: list,
              train_loader,
              device,
              item_id_name='item_id',
              emb_dim=16, c_class_size=19):
        self.model = model
        self.device = device
        assert item_id_name in model.item_id_name, \
            "illegal item id name: {}".format(item_id_name)
        self.item_id_name = item_id_name
        self.item_features = []
        self.output_emb_size = 0
        self.warmup_emb_layer = nn.ModuleDict()
        for item_f in item_features:
            assert item_f in model.features, "unkown feature: {}".format(item_f)
            size, type = self.model.description[item_f]
            if type == 'spr' or type == 'seq':
                self.output_emb_size += emb_dim
                self.warmup_emb_layer["warmup_{}".format(item_f)] = nn.Embedding(size, emb_dim)
            elif type == 'ctn':
                self.output_emb_size += 1
            else:
                raise ValueError('illegal feature tpye for warm: {}'.format(item_f))
            self.item_features.append(item_f)
        self.origin_item_emb = self.model.emb_layer[self.item_id_name]
        self.mean_encoder = nn.Linear(emb_dim, 16)
        self.log_v_encoder = nn.Linear(emb_dim, 16)
        self.mean_encoder_p = nn.Linear(self.output_emb_size, 16)
        self.log_v_encoder_p = nn.Linear(self.output_emb_size, 16)
        self.decoder = nn.Linear(17, 16)

        self.discriminator = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        self.condition_discriminator = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, c_class_size),
            # nn.Sigmoid()
        )

        return

    def wasserstein(self, mean1, log_v1, mean2, log_v2):
        p1 = torch.sum(torch.pow(mean1 - mean2, 2), 1)
        p2 = torch.sum(torch.pow(torch.sqrt(torch.exp(log_v1)) - torch.sqrt(torch.exp(log_v2)), 2), 1)
        return torch.sum(p1 + p2)

    def init_all(self):
        for name, param in self.named_parameters():
            torch.nn.init.uniform_(param, -0.01, 0.01)

    def optimize_all(self):
        for name, param in self.named_parameters():
            param.requires_grad_(True)
        return

    def init_cvaegan(self):
        for name, param in self.named_parameters():
            if ('encoder') in name or ('decoder' in name) or ('discriminator' in name):
                torch.nn.init.uniform_(param, -0.01, 0.01)

    def optimizer_cvaegan(self):
        for name, param in self.named_parameters():
            if ('encoder' in name) or ('decoder' in name) or ('warmup' in name) or ('discriminator' in name):
                param.requires_grad_(True)
            else:
                param.requires_grad_(False)
        return

    def warm_item_id(self, x_dict):
        # get original item id embeddings
        item_ids = x_dict[self.item_id_name]
        item_id_emb = self.origin_item_emb(item_ids).squeeze()
        # get embedding of item features
        item_embs = []
        for item_f in self.item_features:
            type = self.model.description[item_f][1]
            name = "warmup_{}".format(item_f)
            x = x_dict[item_f]
            if type == 'spr':
                emb = self.warmup_emb_layer[name](x).squeeze()
            elif type == 'ctn':
                emb = x
            elif type == 'seq':
                emb = self.warmup_emb_layer[name](x) \
                    .sum(dim=1, keepdim=True).squeeze()
            else:
                raise ValueError('illegal feature tpye for warm: {}'.format(item_f))
            item_embs.append(emb)
        sideinfo_emb = torch.concat(item_embs, dim=1)
        mean = self.mean_encoder(item_id_emb)
        log_v = self.log_v_encoder(item_id_emb)
        mean_p = self.mean_encoder_p(sideinfo_emb)
        log_v_p = self.log_v_encoder_p(sideinfo_emb)
        reg_term = self.wasserstein(mean, log_v, mean_p, log_v_p)
        z = mean + 1e-4 * torch.exp(log_v * 0.5) * torch.randn(mean.size()).to(self.device)
        z_p = mean_p + 1e-4 * torch.exp(log_v_p * 0.5) * torch.randn(mean_p.size()).to(self.device)
        if self.c_feature == 'x_fre':
            freq = x_dict['count']
        elif self.c_feature == 'genres':
            freq = x_dict[self.c_feature][:, [0]]
        else:
            freq = x_dict[self.c_feature]

        pred = self.decoder(torch.concat([z, freq], 1))
        pred_p = self.decoder(torch.concat([z_p, freq], 1))
        recon_term = torch.square(pred - item_id_emb).sum(-1).mean()
        return pred_p, reg_term, recon_term

    def forward(self, x_dict, c_feature='x_fre'):
        warm_id_emb, reg_term, recon_term = self.warm_item_id(x_dict)
        target = self.model.forward_with_item_id_emb(warm_id_emb, x_dict)
        return target, recon_term, reg_term, warm_id_emb

