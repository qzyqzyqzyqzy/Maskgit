import torch
import torch.nn as nn
import torch.nn.functional as F
from mingpt import GPT
from vqgan import VQGAN
import math

class VQGANTransformer(nn.Module):
    def __init__(self, args):
        super(VQGANTransformer, self).__init__()
        self.temperature = args.temperature
        self.top_k = args.top_k
        self.sos_token = args.sos_token
        self.resolution=args.resolution
        self.vqgan = self.load_vqgan(args)

        transformer_config = {
            "vocab_size": args.num_codebook_vectors,
            "block_size": 512,
            "n_layer": 24,
            "n_head": 16,
            "n_embd": 1024
        }
        self.transformer = GPT(**transformer_config)

        self.pkeep = args.pkeep

    @staticmethod
    def load_vqgan(args):
        model = VQGAN(args)
        model.load_checkpoint(args.checkpoint_path)
        model = model.eval()
        return model

    @torch.no_grad()
    def encode_to_z(self, x):
        quant_z, indices, _ = self.vqgan.encode(x)
        indices = indices.view(quant_z.shape[0], -1)
        return quant_z, indices

    @torch.no_grad()
    def z_to_image(self, indices):
        p=(int)(math.sqrt(indices.shape[1]))
        ix_to_vectors = self.vqgan.codebook.embedding(indices).reshape(indices.shape[0], p, p, 512)
        ix_to_vectors = ix_to_vectors.permute(0, 3, 1, 2)
        image = self.vqgan.decode(ix_to_vectors)
        #print("z_to_image_finished")
        return image

    def forward(self, x):
        _, indices = self.encode_to_z(x)
        # print("for ward indices shape")
        # print(indices.shape[1])
        # print(x.shape[1])
        sos_tokens = torch.ones(x.shape[0], 1) * (int)((indices.shape[1])/64)
        sos_tokens = sos_tokens.long().to("cuda")
        #resolution_tokens=torch.ones(x.shape[0], 1) * (x.shape[1]*x.shape[2]*x.shape[3])
        #resolution_tokens = resolution_tokens.long().to("cuda")
        #print(resolution_tokens.shape)
        #sos_tokens=torch.cat((sos_tokens,resolution_tokens),dim=1)
        mask = torch.bernoulli(self.pkeep * torch.ones(indices.shape, device=indices.device))
        mask = mask.round().to(dtype=torch.int64)
        random_indices = torch.randint_like(indices, self.transformer.config.vocab_size)
        new_indices = mask * indices + (1 - mask) * random_indices
        new_indices=torch.cat((sos_tokens,new_indices),dim=1)
        #new_indices = torch.cat((sos_tokens, new_indices), dim=1)

        target = indices

        logits, _ = self.transformer(new_indices[:, :-1])

        return logits, target

    def top_k_logits(self, logits, k):
        v, ix = torch.topk(logits, k)
        out = logits.clone()
        out[out < v[..., [-1]]] = -float("inf")
        return out

    @torch.no_grad()
    def sample(self, x, sos, steps):
        # print("sampleshape")
        # print(x.shape)
        self.transformer.eval()
        sos=sos*(int)(((x.shape[1])+steps)/64)
        x = torch.cat((sos, x), dim=1)
        for k in range(steps):
            logits, _ = self.transformer(x)
            logits = logits[:, -1, :] / self.temperature

            if self.top_k is not None:
                logits = self.top_k_logits(logits, self.top_k)

            probs = F.softmax(logits, dim=-1)

            ix = torch.multinomial(probs, num_samples=1)

            x = torch.cat((x, ix), dim=1)

        x = x[:, sos.shape[1]:]
        self.transformer.train()
        return x

    @torch.no_grad()
    def log_images(self, x):
        log = dict()

        _, indices = self.encode_to_z(x)
        #sos_tokens = torch.ones(x.shape[0], 1) * self.sos_token
        #sos_tokens = sos_tokens.long().to("cuda")
        # print("logimgs indicesshape")
        # print(indices.shape)
        # print(x.shape)
        resolution_tokens=torch.ones(x.shape[0], 1) *(int)((indices.shape[1])/64)
        resolution_tokens = resolution_tokens.long().to("cuda")
        #sos_tokens=torch.cat((sos_tokens,resolution_tokens),dim=1)
        start_indices = indices[:, :indices.shape[1] // 2]
        sample_indices = self.sample(start_indices, resolution_tokens, steps=indices.shape[1] - start_indices.shape[1])
        #print(sample_indices.shape)
        print("half sample finish")
        half_sample = self.z_to_image(sample_indices)
        print(half_sample.shape)
        start_indices = indices[:, :0]
        sample_indices = self.sample(start_indices, resolution_tokens, steps=indices.shape[1])
        full_sample = self.z_to_image(sample_indices)

        x_rec = self.z_to_image(indices)

        log["input"] = x
        log["rec"] = x_rec
        log["half_sample"] = half_sample
        log["full_sample"] = full_sample

        return log, torch.concat((x, x_rec, half_sample, full_sample))
















