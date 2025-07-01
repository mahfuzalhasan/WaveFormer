import torch
import torch.nn as nn
import torch.nn.functional as F


class Contrast(nn.Module):
    """
    Contrastive loss module (e.g., SimCLR NT-Xent style) for two embeddings from
    the same batch. Each batch contains multiple volumes, and each volume
    produces two augmented embeddings. In total, we get 2*N embeddings, where
    each anchor embedding has 1 positive (the other augmentation of the
    same volume) and 2*N - 2 negatives (augmentations of other volumes).
    """
    def __init__(self, args, batch_size, temperature=0.5):
        super().__init__()
        device = torch.device(f"cuda:{args.local_rank}")
        self.batch_size = batch_size
        self.register_buffer("temp", torch.tensor(temperature, device=device))

        # neg_mask: a (2*B, 2*B) boolean matrix with False on diag, True off diag.
        # We'll multiply by that to exclude the anchor's own embedding.
        mask = ~torch.eye(batch_size * 2, dtype=bool)
        self.register_buffer("neg_mask", mask.float().to(device))

    def forward(self, x_i, x_j):
        """
        x_i: [B, embed_dim] embeddings from augmentation 1
        x_j: [B, embed_dim] embeddings from augmentation 2
        Returns a scalar contrastive loss.
        """
        # Normalize embeddings
        z_i = F.normalize(x_i, dim=1)  # [B, embed_dim]
        z_j = F.normalize(x_j, dim=1)  # [B, embed_dim]

        # Concatenate for a total of 2*B embeddings
        z = torch.cat([z_i, z_j], dim=0)  # [2*B, embed_dim]

        # Compute pairwise cosine similarities
        sim = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)
        # sim shape: [2*B, 2*B]

        # Diagonal offsets:
        # sim_ij is the main diagonal offset for i->j pairs
        # sim_ji is the other offset
        # In a standard SimCLR setting, the positive pair for index i is index i+batch_size (for i < B).
        sim_ij = torch.diag(sim, self.batch_size)   # [B]
        sim_ji = torch.diag(sim, -self.batch_size)  # [B]

        # positive pairs: sim_ij, sim_ji
        pos = torch.cat([sim_ij, sim_ji], dim=0)  # shape [2*B]

        # numerator: exp(sim(pos)/temp)
        nom = torch.exp(pos / self.temp)

        # denominator includes all embeddings except self
        # we multiply sim by neg_mask so the diagonal (self-sim) is zeroed out.
        denom = self.neg_mask * torch.exp(sim / self.temp)

        # per-sample loss: -log( nom / sum_over_denominator )
        # sum across all 2*B anchor embeddings
        return torch.sum(-torch.log(nom / torch.sum(denom, dim=1))) / (2 * self.batch_size)


class Loss(nn.Module):
    """
    Combined self-supervised loss for:
      1) Contrastive embeddings from two augmentations
      2) Reconstruction of the original volume from each augmented version

    This version no longer contains a rotation head or rotation loss.
    """
    def __init__(self, batch_size, args):
        super().__init__()
        self.contrast_loss = Contrast(args, batch_size).cuda()
        self.recon_loss = nn.L1Loss().cuda()

        # Weighting factors (can be tuned)
        self.alpha_contrast = 1.0
        self.alpha_recon = 1.0

    def __call__(self, c1, c2, rec1, rec2, gt1, gt2):
        """
        Args:
            c1, c2:   [B, embedding_dim] contrastive embeddings for the two augmentations
            rec1, rec2: [B, C, H, W, D] volumes reconstructed from augmentation1, augmentation2
            gt1, gt2:   [B, C, H, W, D] original volumes (same as x in your script), 
                        used as reconstruction targets for rec1, rec2

        Returns:
            total_loss: a scalar
            (contrast_loss, recon_loss): the 2 main components
        """

        # 1) Contrastive loss: c1 vs c2
        contrast_l = self.contrast_loss(c1, c2)

        # 2) Reconstruction loss: rec1 vs gt1, rec2 vs gt2
        rec_l1 = self.recon_loss(rec1, gt1)
        rec_l2 = self.recon_loss(rec2, gt2)
        recon_l = 0.5 * (rec_l1 + rec_l2)  # average them

        total_loss = self.alpha_contrast * contrast_l * recon_l + self.alpha_recon * recon_l

        return total_loss, (contrast_l, recon_l)
