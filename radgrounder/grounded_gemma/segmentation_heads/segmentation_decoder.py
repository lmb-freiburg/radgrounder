import torch
from torch import nn
from torch.nn import functional as F
import math
from typing import Tuple

class TwoWayAttentionBlock(nn.Module):
    """
    A transformer block with four layers similar to SAM:
    (1) self-attention of sparse inputs, 
    (2) cross attention of sparse inputs to dense inputs, 
    (3) mlp block on sparse inputs, 
    (4) cross attention of dense inputs to sparse inputs.
    """
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int = 2048,
        skip_first_layer_pe: bool = False,
    ) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)

        self.cross_attn_token_to_image = MultiHeadAttention(embedding_dim, num_heads)
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, embedding_dim),
        )
        self.norm3 = nn.LayerNorm(embedding_dim)

        self.norm4 = nn.LayerNorm(embedding_dim)
        self.cross_attn_image_to_token = MultiHeadAttention(embedding_dim, num_heads)

        self.skip_first_layer_pe = skip_first_layer_pe

    def forward(
        self, queries: torch.Tensor, keys: torch.Tensor, query_pe: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self attention block
        if self.skip_first_layer_pe or query_pe is None:
            attn_out = self.self_attn(queries, queries, queries)
        else:
            q = queries + query_pe
            attn_out = self.self_attn(q, q, queries)
        queries = queries + attn_out
        queries = self.norm1(queries)

        # Cross attention block, tokens attending to image embedding
        q = queries + query_pe
        attn_out = self.cross_attn_token_to_image(q, keys, keys)
        queries = queries + attn_out
        queries = self.norm2(queries)

        # MLP block
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)

        # Cross attention block, image embedding attending to tokens
        q = queries + query_pe
        attn_out = self.cross_attn_image_to_token(keys, q, queries)
        keys = keys + attn_out
        keys = self.norm4(keys)

        return queries, keys


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention module similar to SAM's implementation.
    """
    def __init__(self, embedding_dim: int, num_heads: int) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        assert embedding_dim % num_heads == 0, "embedding_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embedding_dim, embedding_dim)
        self.k_proj = nn.Linear(embedding_dim, embedding_dim)
        self.v_proj = nn.Linear(embedding_dim, embedding_dim)
        self.out_proj = nn.Linear(embedding_dim, embedding_dim)

    def _separate_heads(self, x: torch.Tensor, num_heads: int) -> torch.Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Attention
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
        attn = attn / math.sqrt(c_per_head)
        attn = torch.softmax(attn, dim=-1)

        # Get output
        out = attn @ v
        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out


class MultiLayerTwoWayTransformer(nn.Module):
    """
    Multi-layer transformer decoder similar to SAM's mask decoder.
    """
    def __init__(
        self,
        depth: int,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int,
    ) -> None:
        super().__init__()
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.layers = nn.ModuleList()

        for i in range(depth):
            self.layers.append(
                TwoWayAttentionBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    skip_first_layer_pe=(i == 0),
                )
            )

        self.final_attn_token_to_image = MultiHeadAttention(embedding_dim, num_heads)
        self.norm_final_attn = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        point_embedding: torch.Tensor,
        image_embedding: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
          point_embedding (torch.Tensor): the embedding to add to the query points.
            Must have shape B x N_points x embedding_dim for any N_points.
          image_embedding (torch.Tensor): image tokens to attend to. Should be shape
            B x N_image_tokens x embedding_dim.

        Returns:
          torch.Tensor: the processed point_embedding
          torch.Tensor: the processed image_embedding
        """
        # Prepare queries
        queries = point_embedding
        keys = image_embedding

        # Apply transformer blocks and final layernorm
        for layer in self.layers:
            queries, keys = layer(
                queries=queries,
                keys=keys,
                query_pe=point_embedding,  # Use original point embedding as positional encoding
            )

        # Apply the final attention layer from the points to the image
        attn_out = self.final_attn_token_to_image(queries, keys, keys)
        queries = queries + attn_out
        queries = self.norm_final_attn(queries)

        return queries, keys


class SegmentationDecoder(nn.Module):
    def __init__(self, img_feat_dim, proj_feat_dim, num_classes=1, num_heads=8, num_layers=2):
        super().__init__()
        self.proj_feat_dim = proj_feat_dim
        self.num_classes = num_classes
        self.spatial_dim = (16, 16)
        self.num_heads = num_heads
        self.num_layers = num_layers

        # Project image features
        self.img_projection = nn.Linear(img_feat_dim, proj_feat_dim)

        # Multi-layer Two-Way Transformer similar to SAM
        self.transformer = MultiLayerTwoWayTransformer(
            depth=num_layers,
            embedding_dim=proj_feat_dim,
            num_heads=num_heads,
            mlp_dim=proj_feat_dim * 4,
        )

        # Project fused image features into per-patch class logits
        self.output_proj = nn.Linear(proj_feat_dim, num_classes)
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.MultiheadAttention):
            nn.init.xavier_uniform_(module.in_proj_weight)
            nn.init.zeros_(module.in_proj_bias)
            nn.init.xavier_uniform_(module.out_proj.weight)
            nn.init.zeros_(module.out_proj.bias)
        elif isinstance(module, MultiHeadAttention):
            # Initialize custom MultiHeadAttention weights
            nn.init.xavier_uniform_(module.q_proj.weight)
            nn.init.zeros_(module.q_proj.bias)
            nn.init.xavier_uniform_(module.k_proj.weight)
            nn.init.zeros_(module.k_proj.bias)
            nn.init.xavier_uniform_(module.v_proj.weight)
            nn.init.zeros_(module.v_proj.bias)
            nn.init.xavier_uniform_(module.out_proj.weight)
            nn.init.zeros_(module.out_proj.bias)
        elif isinstance(module, TwoWayAttentionBlock):
            # Let recursion handle submodules; nothing to do here
            pass
        elif isinstance(module, MultiLayerTwoWayTransformer):
            # Let recursion handle submodules; nothing to do here
            pass
        elif isinstance(module, nn.Sequential) or isinstance(module, nn.ModuleList):
            # Containers, nothing to do
            pass
        else:
            # Only warn for truly unhandled modules
            print(f"Warning: Unhandled module type {type(module)} in _init_weights")

    def forward(self, image_feats, p_end_embed):
        """
        image_feats: (B, 256, img_feat_dim) - from SigLIP or ViT (16x16 tokens)
        p_end_embed: (B, proj_feat_dim) - embedding of </p> or similar
        """
        B, N, _ = image_feats.shape
        H, W = self.spatial_dim
        
        img_proj = self.img_projection(image_feats)  # (B, 256, D)

        # Prompt embedding as query
        query = p_end_embed.unsqueeze(1)  # (B, 1, D)

        # Apply multi-layer two-way transformer similar to SAM
        # This performs bidirectional attention between prompt tokens and image tokens
        decoded_queries, enhanced_keys = self.transformer(query, img_proj)
        
        # Use the enhanced image features for final prediction
        # Fuse decoded token with enhanced image tokens (broadcast)
        fused = enhanced_keys + decoded_queries.expand(-1, N, -1)  # (B, 256, D)

        # Project fused tokens to class logits
        logits = self.output_proj(fused)  # (B, 256, C)

        # Reshape to (B, C, H, W)
        mask_logits = logits.transpose(1, 2).reshape(B, self.num_classes, H, W)
        # print("Mask logits shape before upsampling:", mask_logits.shape)

        final_mask = F.interpolate(mask_logits, size=(224, 224), mode='bilinear', align_corners=False)
        return final_mask
    
if __name__ == "__main__":
    B = 5  # batch size
    C = 1  # number of classes
    H, W = 16, 16  # token grid (e.g. 16x16 = 256 tokens)
    img_feat_dim = 1152  # feature dim from image encoder
    proj_feat_dim = 2304  # transformer dim

    # Random image features from SigLIP or ViT
    image_feats = torch.randn(B, H * W, img_feat_dim) * 70

    # Random prompt token (</p>) embedding
    p_end_embed = torch.randn(B, proj_feat_dim) * 70

    # Model with multi-layer attention (similar to SAM)
    model = SegmentationDecoder(
        img_feat_dim=img_feat_dim,
        proj_feat_dim=proj_feat_dim,
        num_classes=C,
        num_heads=8,  # Increased number of heads
        num_layers=2  # Number of transformer layers
    )

    # Forward pass
    mask = model(image_feats, p_end_embed)
    print("Mask shape:", mask.shape)  # Expected: (B, C, 224, 224)
    
    gt_mask = torch.randn(B, C, 224, 224)  # Example ground truth mask for testing
    
    bce_loss = nn.BCEWithLogitsLoss()
    loss = bce_loss(mask, gt_mask)
    print("Loss:", loss.item())
    
    # Test with different configurations
    print("\nTesting different configurations:")
    
    # Test with more layers
    model_deep = SegmentationDecoder(
        img_feat_dim=img_feat_dim,
        proj_feat_dim=proj_feat_dim,
        num_classes=C,
        num_heads=8,
        num_layers=4  # Deeper model
    )
    
    mask_deep = model_deep(image_feats, p_end_embed)
    print("Deep model mask shape:", mask_deep.shape)
    
    # Test with more heads
    model_more_heads = SegmentationDecoder(
        img_feat_dim=img_feat_dim,
        proj_feat_dim=proj_feat_dim,
        num_classes=C,
        num_heads=16,  # More attention heads
        num_layers=2
    )
    
    mask_more_heads = model_more_heads(image_feats, p_end_embed)
    print("More heads model mask shape:", mask_more_heads.shape)