# Note: the file https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py#L519 
# is the base file for the custom model, every change is marked by ##



# every original import needed, copied from 
# https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py#L519
from functools import partial
from typing import Callable, Optional, Tuple, Type, Union
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

 
from timm.layers import PatchEmbed, Mlp, AttentionPoolLatent, PatchDropout, \
    get_act_layer, get_norm_layer, LayerType

# import classes needed for the custom model
from timm.models.vision_transformer import VisionTransformer, Block, Attention



class MyCustomAttention(Attention):
    """ Custom attention class that saves the attention map for the class token only """
    def __init__(self, save_attn, *args, **kwargs):
        """ Initialize the custom attention class

        Args:
            save_attn (bool): Flag to save the attention maps
        """
        super().__init__(*args, **kwargs)

        ## add object and extra flag for saving attention maps 
        self.attn_map = None
        self.save_attn = save_attn 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Forward pass of the custom attention layer 
        
        Args:
            x (torch.Tensor): Input tensor
        
        Returns:
            torch.Tensor: Output tensor of forward pass
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)


        ## changed the code to save the attention maps 
        if not self.fused_attn or self.save_attn: # to ensure that fused attention is not used
            # print("Normal attention")
            q = q * self.scale
            ## Only look at the class token
            # Only use the query for the class token (index 0)
            q = q[:, :, 0, :]  # Shape: (B, num_heads, C//num_heads)
            q = q.unsqueeze(2)  # Shape: (B, num_heads, 1, C//num_heads) such that it can be multiplied with k
            # Use the key for all patches (exclude class token from key and value)
            k = k[:, :, 1:, :]  # Shape: (B, num_heads, N-1, C//num_heads)
            v = v[:, :, 1:, :]  # Shape: (B, num_heads, N-1, C//num_heads)

            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            
            # print("Saving attention maps")
            self.attn_map = attn

            x = attn @ v
            ## slightly differing dimensions
            x = x.transpose(1, 2).reshape(B, 1, C)
            
        else:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
            x = x.transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class MyCustomBlock(Block):
    """ Custom block class that uses the custom attention class """
    def __init__(self, save_attn, *args, **kwargs):
        """ Initialize the custom block class

        Args:
            save_attn (bool): Flag to save the attention maps
        """
        super().__init__(*args, **kwargs)
        
        ## get the arguments from the original Block class needed to initialize the custom attention
        dim = kwargs.get('dim')
        num_heads = kwargs.get('num_heads')
        qkv_bias = kwargs.get('qkv_bias', False)
        qk_norm = kwargs.get('qk_norm', False)
        attn_drop = kwargs.get('attn_drop', 0.)
        proj_drop = kwargs.get('proj_drop', 0.)
        norm_layer = kwargs.get('norm_layer', nn.LayerNorm)
        
        ## initialize the custom attention with save_attn argument
        self.attn = MyCustomAttention(
            save_attn,
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )


class MyCustomVisionTransformer(VisionTransformer):
    """ Custom Vision Transformer class that uses the custom block and attention
    classes """
    def __init__(
            self,
            img_size: Union[int, Tuple[int, int]] = 224,
            patch_size: Union[int, Tuple[int, int]] = 16,
            in_chans: int = 3,
            num_classes: int = 1000,
            global_pool: Literal['', 'avg', 'avgmax', 'max', 'token', 'map'] = 'token',
            embed_dim: int = 768,
            depth: int = 12,
            num_heads: int = 12,
            mlp_ratio: float = 4.,
            qkv_bias: bool = True,
            qk_norm: bool = False,
            init_values: Optional[float] = None,
            class_token: bool = True,
            pos_embed: str = 'learn',
            no_embed_class: bool = False,
            reg_tokens: int = 0,
            pre_norm: bool = False,
            fc_norm: Optional[bool] = None,
            dynamic_img_size: bool = False,
            dynamic_img_pad: bool = False,
            drop_rate: float = 0.,
            pos_drop_rate: float = 0.,
            patch_drop_rate: float = 0.,
            proj_drop_rate: float = 0.,
            attn_drop_rate: float = 0.,
            drop_path_rate: float = 0.,
            weight_init: Literal['skip', 'jax', 'jax_nlhb', 'moco', ''] = '',
            fix_init: bool = False,
            embed_layer: Callable = PatchEmbed,
            norm_layer: Optional[LayerType] = None,
            act_layer: Optional[LayerType] = None,
            block_fn: Type[nn.Module] = MyCustomBlock, ## changed the block_fn to the custom block!
            mlp_layer: Type[nn.Module] = Mlp,
    ) -> None:
        """
        Args:
            img_size: Input image size.
            patch_size: Patch size.
            in_chans: Number of image input channels.
            num_classes: Number of classes for classification head.
            global_pool: Type of global pooling for final sequence (default: 'token').
            embed_dim: Transformer embedding dimension.
            depth: Depth of transformer.
            num_heads: Number of attention heads.
            mlp_ratio: Ratio of mlp hidden dim to embedding dim.
            qkv_bias: Enable bias for qkv projections if True.
            init_values: Layer-scale init values (layer-scale enabled if not None).
            class_token: Use class token.
            no_embed_class: Don't include position embeddings for class (or reg) tokens.
            reg_tokens: Number of register tokens.
            fc_norm: Pre head norm after pool (instead of before), if None, enabled when global_pool == 'avg'.
            drop_rate: Head dropout rate.
            pos_drop_rate: Position embedding dropout rate.
            attn_drop_rate: Attention dropout rate.
            drop_path_rate: Stochastic depth rate.
            weight_init: Weight initialization scheme.
            fix_init: Apply weight initialization fix (scaling w/ layer index).
            embed_layer: Patch embedding layer.
            norm_layer: Normalization layer.
            act_layer: MLP activation layer.
            block_fn: Transformer block layer.
        """
        super().__init__()
        assert global_pool in ('', 'avg', 'avgmax', 'max', 'token', 'map')
        assert class_token or global_pool != 'token'
        assert pos_embed in ('', 'none', 'learn')
        use_fc_norm = global_pool in ('avg', 'avgmax', 'max') if fc_norm is None else fc_norm
        norm_layer = get_norm_layer(norm_layer) or partial(nn.LayerNorm, eps=1e-6)
        act_layer = get_act_layer(act_layer) or nn.GELU

        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.head_hidden_size = self.embed_dim = embed_dim  # for consistency with other models
        self.num_prefix_tokens = 1 if class_token else 0
        self.num_prefix_tokens += reg_tokens
        self.num_reg_tokens = reg_tokens
        self.has_class_token = class_token
        self.no_embed_class = no_embed_class  # don't embed prefix positions (includes reg)
        self.dynamic_img_size = dynamic_img_size
        self.grad_checkpointing = False

        embed_args = {}
        if dynamic_img_size:
            # flatten deferred until after pos embed
            embed_args.update(dict(strict_img_size=False, output_fmt='NHWC'))
        self.patch_embed = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            bias=not pre_norm,  # disable bias if pre-norm is used (e.g. CLIP)
            dynamic_img_pad=dynamic_img_pad,
            **embed_args,
        )
        num_patches = self.patch_embed.num_patches
        reduction = self.patch_embed.feat_ratio() if hasattr(self.patch_embed, 'feat_ratio') else patch_size

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if class_token else None
        self.reg_token = nn.Parameter(torch.zeros(1, reg_tokens, embed_dim)) if reg_tokens else None
        embed_len = num_patches if no_embed_class else num_patches + self.num_prefix_tokens
        if not pos_embed or pos_embed == 'none':
            self.pos_embed = None
        else:
            self.pos_embed = nn.Parameter(torch.randn(1, embed_len, embed_dim) * .02)
        self.pos_drop = nn.Dropout(p=pos_drop_rate)
        if patch_drop_rate > 0:
            self.patch_drop = PatchDropout(
                patch_drop_rate,
                num_prefix_tokens=self.num_prefix_tokens,
            )
        else:
            self.patch_drop = nn.Identity()
        self.norm_pre = norm_layer(embed_dim) if pre_norm else nn.Identity()

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                init_values=init_values,
                proj_drop=proj_drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                mlp_layer=mlp_layer,
                save_attn=True if i == depth - 1 else False, ## changed/added the save_attn to True for the last block
            )
            for i in range(depth)])
        self.feature_info = [
            dict(module=f'blocks.{i}', num_chs=embed_dim, reduction=reduction) for i in range(depth)]
        self.norm = norm_layer(embed_dim) if not use_fc_norm else nn.Identity()

        # Classifier Head
        if global_pool == 'map':
            self.attn_pool = AttentionPoolLatent(
                self.embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                norm_layer=norm_layer,
            )
        else:
            self.attn_pool = None
        self.fc_norm = norm_layer(embed_dim) if use_fc_norm else nn.Identity()
        self.head_drop = nn.Dropout(drop_rate)
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        if weight_init != 'skip':
            self.init_weights(weight_init)
        if fix_init:
            self.fix_init_weight()


        
################################################ Functions for testing the custom ViT ################################################     

import os
import timm
import argparse

from tqdm import tqdm


# docker
"""
screen -dmS custom_vit sh -c 'docker run --shm-size=400gb --gpus \"device=0\" --name ftoelkes_run0 -it -u `id -u $USER` --rm -v /home/ftoelkes/code/lymphoma:/mnt -v /home/ftoelkes/code/lymphoma/vit_large_patch16_256.dinov2.uni_mass100k:/model ftoelkes_lymphoma python3 -m data_preparation.custom_vision_transformer --patch_size=1024; exec bash'
"""



def parse_args() -> dict:
    """Parse command line arguments
    
    Returns:
        dict: Arguments from command line
    """
    parser = argparse.ArgumentParser('Custom Vision Transformer')
    parser.add_argument('--patch_size', default=224, type=int, help='Patch size (default: 224)')
    args = parser.parse_args()
    return dict(vars(args))

    
def load_custom_model(device: torch.device, local_dir: str = "../vit_large_patch16_256.dinov2.uni_mass100k") -> MyCustomVisionTransformer:
    """Load the model modified with subclassing.
    
    Args:
        device (torch.device): Device to load the model on
        local_dir (str): Local directory where the model is saved (default: "../vit_large_patch16_256.dinov2.uni_mass100k")

    Returns:
        MyCustomVisionTransformer: Custom model
    """
    # use the parameters from p.56 of UNI paper (https://arxiv.org/pdf/2308.15474)
    model = MyCustomVisionTransformer(img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True,
                                      embed_dim=1024, depth=24, num_heads=16)
    model.load_state_dict(torch.load(os.path.join(local_dir, "model.pth"), map_location="cpu"), strict=True)
    model.to(device)
    model.eval()
    return model


def load_wrapper_model(device: torch.device, local_dir: str) -> timm.models.vision_transformer.VisionTransformer:
    """Load the model modified with a wrapper function.
    
    Args:
        device (torch.device): Device to load the model on
        local_dir (str): Local directory where the model is saved

    Returns:
        timm.models.vision_transformer.VisionTransformer: Wrapper model
    """
    model = timm.create_model("vit_large_patch16_224", img_size=224, patch_size=16, 
                              init_values=1e-5, num_classes=0, dynamic_img_size=True)
    model.load_state_dict(torch.load(os.path.join(local_dir, "model.pth"), map_location="cpu"), strict=True)
    model.to(device)
    model.eval()
    model.blocks[-1].attn.forward = forward_wrapper(model.blocks[-1].attn)
    return model

def forward_wrapper(attn_obj: Attention) -> Callable:
    """Wrapper function to save the attention map for the class token only

    Args:
        attn_obj (Attention): Attention object

    Returns:
        Callable: Wrapper function for forward pass
    """
    def attn_saving_forward(x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the wrapper function

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output tensor of forward pass
        """
        B, N, C = x.shape # B: batch size, N: number of 16x16 patches, C: channels
        nh = attn_obj.num_heads
        qkv = attn_obj.qkv(x).reshape(B, N, 3, nh, C // nh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0) # make torchscript happy (cannot use tensor as tuple)

        # Only use the query for the class token (index 0)
        q = q[:, :, 0, :]  # Shape: (B, num_heads, C//num_heads)
        q = q.unsqueeze(2)  # Shape: (B, num_heads, 1, C//num_heads) such that it can be multiplied with k

        # Use the key for all patches (exclude class token from key and value)
        k = k[:, :, 1:, :]  # Shape: (B, num_heads, N-1, C//num_heads)
        v = v[:, :, 1:, :]  # Shape: (B, num_heads, N-1, C//num_heads)

        # Instead of computing the full attention map, we just compute the class-to-patch attention
        attn = (q @ k.transpose(-2, -1)) * attn_obj.scale  
        attn = attn.softmax(dim=-1) # Shape: (B, num_heads, num_heads, N-1)
        attn = attn_obj.attn_drop(attn) # not important since we only use it in inference

        # Save the attention map for class token only 
        attn_obj.attn_map = attn 

        # Compute the output using the value vectors of the patches
        # We only need to apply attention to v_patches, skipping the class token's self-attention
        x = (attn @ v).transpose(1, 2).reshape(B, 1, C) 
        x = attn_obj.proj(x)
        x = attn_obj.proj_drop(x) # not important since we only use it in inference
        return x
    return attn_saving_forward


def main(patch_size: int):
    """Test the custom model with random input
    
    Args:
        patch_size (int): Patch size
    """
    from .generate_uni_embeddings import load_uni_model # avoid circular import

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    local_dir = "/model" if torch.cuda.is_available() else "../vit_large_patch16_256.dinov2.uni_mass100k"

    model = load_uni_model(device, local_dir)
    custom_model = load_custom_model(device, local_dir)
    wrapper_model = load_wrapper_model(device, local_dir)  

    # obtain outputs and compute difference for random inputs
    cosine_similarity = [0,0,0]
    mse = [0,0,0]
    n = 1
    # set seed for reproducibility
    torch.manual_seed(0)
    with torch.no_grad():
        for _ in tqdm(range(n), desc="Testing"):
            x = torch.randn(1, 3, patch_size, patch_size).to(device)
            y = model(x)
            custom_y = custom_model(x)
            wrapper_y = wrapper_model(x)

            cosine_similarity[0] += F.cosine_similarity(y, custom_y).item()
            cosine_similarity[1] += F.cosine_similarity(y, wrapper_y).item()
            cosine_similarity[2] += F.cosine_similarity(wrapper_y, custom_y).item()
            mse[0] += F.mse_loss(y, custom_y).item()
            mse[1] += F.mse_loss(y, wrapper_y).item()
            mse[2] += F.mse_loss(wrapper_y, custom_y).item()

    print(f"Cosine similarity: {cosine_similarity[0]/n}, {cosine_similarity[1]/n}, {cosine_similarity[2]/n}")
    print(f"MSE: {mse[0]/n}, {mse[1]/n}, {mse[2]/n}")

        
if __name__ == "__main__":
    args = parse_args()
    main(**args)






