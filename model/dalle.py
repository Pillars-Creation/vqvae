import torch
import torch.nn as nn
import torch.nn.functional as F


class DALLE(nn.Module):
    def __init__(self,
                 model: nn.Module,
                 warm_features: list,
                 train_loader,
                 device,
                 item_id_name = 'item_id',
                 emb_dim = 16):
        super(DALLE, self).__init__()
        self.codebook_size = 32
        self.codebook_dim = 8
        self.codebook = nn.Parameter(torch.randn(self.codebook_size, self.codebook_dim))
        self.commitment_cost = 0.25
        self.build(model, warm_features, train_loader, device, item_id_name, emb_dim)

        # multi head
        self.emb_dim = 8
        self.num_heads = 4
        self.head_dim = self.emb_dim // self.num_heads
        self.loop_number = 1
        assert self.head_dim * self.num_heads == self.emb_dim, "emb_dim must be divisible by num_heads"

        self.query_linear = nn.Linear(self.emb_dim, self.emb_dim)
        self.key_linear = nn.Linear(self.emb_dim, self.emb_dim)
        self.value_linear = nn.Linear(self.emb_dim, self.emb_dim)
        self.fc_out = nn.Linear(self.emb_dim, emb_dim)

        return

    def scaled_dot_product_attention(self, query, key, value, mask=None):
        seq_length = query.size(-2)
        dot_product = torch.matmul(query, key.transpose(-2, -1))
        scaled_dot_product = dot_product / (self.head_dim ** 0.5)

        if mask is not None:
            scaled_dot_product = scaled_dot_product.masked_fill(mask == 0, float("-inf"))

        attention = F.softmax(scaled_dot_product, dim=-1)
        output = torch.matmul(attention, value)
        return output, attention

    def multi_head_attention(self, x, y="self", N=1):
        """
        Multi-head attention for input sequence.

        Args:
            x: input sequence tensor, shape [batch_size, seq_len, emb_dim].
            y: target sequence tensor, shape [batch_size, seq_len, emb_dim] or "self".
            N: number of iterations to run the multi-head attention.

        Returns:
            output tensor, shape [batch_size, emb_dim].
        """
        batch_size = x.size(0)

        for _ in range(N):
            if y == "self":
                query = self.query_linear(x)
                key = self.key_linear(x)
                value = self.value_linear(x)
            else:
                query = x
                key = y
                value = y
                query = self.query_linear(query)
                key = self.key_linear(key)
                value = self.value_linear(value)

            query = query.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            key = key.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            value = value.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

            output, attention = self.scaled_dot_product_attention(query, key, value)
            output = output.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.emb_dim)

            # Update input for the next iteration
            x = output

        return output

    def build(self,
              model: nn.Module,
              item_features: list,
              train_loader,
              device,
              item_id_name = 'item_id',
              emb_dim = 16):
        self.model = model
        self.device = device
        assert item_id_name in model.item_id_name, \
                        "illegal item id name: {}".format(item_id_name)
        self.item_id_name = item_id_name
        self.item_features = []
        self.output_emb_size = 0
        for item_f in item_features:
            assert item_f in model.features, "unkown feature: {}".format(item_f)
            type = self.model.description[item_f][1]
            if type == 'spr' or type == 'seq':
                self.output_emb_size += emb_dim
            elif type == 'ctn':
                self.output_emb_size += 1
            else:
                raise ValueError('illegal feature tpye for warm: {}'.format(item_f))
            self.item_features.append(item_f)
        self.origin_item_emb = self.model.emb_layer[self.item_id_name]

        self.mean_encoder = nn.Sequential(
            nn.Linear(emb_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
        )

        self.log_v_encoder = nn.Sequential(
            nn.Linear(emb_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
        )

        self.mean_encoder_p = nn.Sequential(
            nn.Linear(self.output_emb_size, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
        )

        self.log_v_encoder_p = nn.Sequential(
            nn.Linear(self.output_emb_size, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
        )

        self.decoder = nn.Sequential(
            nn.Linear(9, 12),
            nn.ReLU(),
            nn.Linear(12, 16),
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

    def init_dalle(self):
        for name, param in self.named_parameters():
            if ('encoder') in name or ('decoder' in name) :
                torch.nn.init.uniform_(param, -0.01, 0.01)

    def optimizer_dalle(self):
        for name, param in self.named_parameters():
            if ('encoder' in name) or ('decoder' in name) :
                param.requires_grad_(True)
            else:
                param.requires_grad_(False)
            for item_f in self.item_features:
                if item_f in name:
                    param.requires_grad_(True)
        return

    def optimizer_vae(self):
        for name, param in self.named_parameters():
            if ('encoder' in name) or ('decoder' in name) or ('query_linear' in name) \
                    or ('key_linear' in name) or ('value_linear' in name) or ('fc_out' in name) :
                param.requires_grad_(True)
            else:
                param.requires_grad_(False)
            for item_f in self.item_features:
                if item_f in name:
                    param.requires_grad_(True)
        return

    def optimizer_codebook(self):
        for name, param in self.named_parameters():
            if ('codebook' in name):
                param.requires_grad_(True)
            else:
                param.requires_grad_(False)
        return

    def vector_quantizer(self, z):
        # 将 z 的形状更改为 [batch_size, embedding_dim, 1]
        z_flat = z.view(-1, self.codebook_dim, 1)

        # 计算 z_flat 两两相乘的结果
        z_flat = torch.matmul(z_flat, z_flat.transpose(1, 2))
        z_flat = torch.sqrt(z_flat)

        # 计算z_flat中每个潜在向量与码本中所有向量之间的欧几里得距离
        distances = torch.cdist(z_flat, self.codebook)

        # 计算与每个潜在向量z最接近的码本向量的索引
        codebook_indices = torch.argmin(distances, dim=-1)

        # 使用codebook_indices从码本中检索与原始潜在向量z最接近的离散潜在向量z_q
        one_hot = F.one_hot(codebook_indices, self.codebook_size).type(z_flat.dtype)
        z_q = torch.matmul(one_hot, self.codebook)

        # 提取 z_q 的对角线元素并将它们相加以还原为形状为 [batch_size, emb] 的张量
        z_q = torch.diagonal(z_q, dim1=1, dim2=2)


        # 计算VQ损失，vq_loss为标量
        vq_loss = torch.mean(torch.square(z_q.detach() - z))
        commit_loss = torch.mean(torch.square(z.detach() - z_q))
        vq_loss += self.commitment_cost * commit_loss

        # Apply the Straight-Through Estimator (STE) trick
        z_q = z + (z_q - z).detach()

        # 计算困惑度
        avg_probs = torch.mean(one_hot, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # VQ-VAE Decoder
        z_q = z_q.view(z.shape)

        return z_q, vq_loss, perplexity


    def warm_item_id(self, x_dict):
        # get original item id embeddings
        item_ids = x_dict[self.item_id_name]
        item_id_emb = self.origin_item_emb(item_ids).squeeze()
        # get embedding of item features
        item_embs = []
        for item_f in self.item_features:
            type = self.model.description[item_f][1]
            x = x_dict[item_f]
            if type == 'spr':
                emb = self.model.emb_layer[item_f](x).squeeze()
            elif type == 'ctn':
                emb = x
            elif type == 'seq':
                emb = self.model.emb_layer[item_f](x) \
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
        z = mean + torch.exp(log_v * 0.5) * torch.randn(mean.size()).to(self.device)
        z_p = mean_p + torch.exp(log_v_p * 0.5) * torch.randn(mean_p.size()).to(self.device)
        z_q , vq_loss, perplexity = self.vector_quantizer(z)
        freq = x_dict['count']
        decode_out = self.decoder(torch.concat([z_q, freq], 1))
        recon_loss = torch.square(decode_out - item_id_emb).sum(-1).mean()

        codebook_expanded = self.codebook.unsqueeze(0).expand(z_p.size(0), -1, -1)
        z_p = z_p.view(-1, self.codebook_dim, 1)
        z_p = torch.matmul(z_p, z_p.transpose(1, 2))
        # z_p = torch.sqrt(z_p)
        concatenated = torch.cat([z_p, codebook_expanded], dim=1)
        pred = self.multi_head_attention(concatenated, 'self', self.loop_number)
        pred = pred.sum(dim=1)
        pred = self.fc_out(pred)
        return pred, reg_term, vq_loss, perplexity, recon_loss

    def warm_item_id_p(self, x_dict):
        # get embedding of item features
        item_embs = []
        for item_f in self.item_features:
            type = self.model.description[item_f][1]
            x = x_dict[item_f]
            if type == 'spr':
                emb = self.model.emb_layer[item_f](x).squeeze()
            elif type == 'ctn':
                emb = x
            elif type == 'seq':
                emb = self.model.emb_layer[item_f](x) \
                        .sum(dim=1, keepdim=True).squeeze()
            else:
                raise ValueError('illegal feature tpye for warm: {}'.format(item_f))
            item_embs.append(emb)
        sideinfo_emb = torch.concat(item_embs, dim=1)
        freq = x_dict['count']
        mean_p = self.mean_encoder_p(sideinfo_emb)
        log_v_p = self.log_v_encoder_p(sideinfo_emb)
        z = mean_p + torch.exp(log_v_p * 0.5) * torch.randn(mean_p.size()).to(self.device)
        z_p = mean_p + torch.exp(log_v_p * 0.5) * torch.randn(mean_p.size()).to(self.device)
        z_q , vq_loss, perplexity = self.vector_quantizer(z)
        # pred = self.decoder(torch.concat([z_q, freq], 1))
        codebook_expanded = self.codebook.unsqueeze(0).expand(z_p.size(0), -1, -1)
        z_p = z_p.view(-1, self.codebook_dim, 1)
        z_p = torch.matmul(z_p, z_p.transpose(1, 2))
        # z_p = torch.sqrt(z_p)
        concatenated = torch.cat([z_p, codebook_expanded], dim=1)
        pred = self.multi_head_attention(concatenated, 'self', self.loop_number)
        pred = pred.sum(dim=1)
        pred = self.fc_out(pred)

        return pred


    def forward(self, x_dict):
        item_ids = x_dict[self.item_id_name]
        item_id_emb = self.origin_item_emb(item_ids).squeeze()
        warm_id_emb, reg_term, vq_loss, perplexity, recon_loss = self.warm_item_id(x_dict)
        target = self.model.forward_with_item_id_emb(warm_id_emb, x_dict)
        return recon_loss, reg_term, target, vq_loss, perplexity