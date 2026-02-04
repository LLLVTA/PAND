"""
CoOp (Context Optimization) implementation for Lightning + open_clip.

Adapted from original CoOp paper: https://arxiv.org/abs/2109.01134
GitHub: https://github.com/KaiyangZhou/CoOp

Key design decisions:
1. Only support class_token_position='end' (not middle/front) for stability
2. Use "X X X" placeholder for prompt template construction (even with ctx_init)
3. Fixed context_length (77) for stable tokenization and embedding concatenation
4. Freeze all parameters except prompt_learner.ctx
5. Production-grade error handling and multi-GPU compatibility
"""

import torch
import torch.nn as nn
from typing import List, Optional
import open_clip


def get_sot_token_id(tokenizer) -> int:
    """
    Retrieve Start-of-Text (SoT/BOS) token ID from tokenizer.
    
    Args:
        tokenizer: open_clip tokenizer
        
    Returns:
        int: SoT token ID
        
    Raises:
        AttributeError: If tokenizer doesn't have recognizable SoT token
    """
    if hasattr(tokenizer, 'sot_token_id'):
        return tokenizer.sot_token_id
    elif hasattr(tokenizer, 'bos_token_id'):
        return tokenizer.bos_token_id
    elif hasattr(tokenizer, 'start_token_id'):
        return tokenizer.start_token_id
    else:
        # Fallback for CLIP SimpleTokenizer (classic CLIP BPE)
        tokenizer_type = type(tokenizer).__name__
        if 'SimpleTokenizer' in tokenizer_type or 'CLIPTokenizer' in tokenizer_type:
            return 49406  # Standard CLIP SOT token ID
        raise AttributeError(
            f"Tokenizer {tokenizer_type} does not have "
            "sot_token_id / bos_token_id / start_token_id attribute. "
            "If using custom tokenizer, please add sot_token_id attribute."
        )


def get_eot_token_id(tokenizer) -> int:
    """
    Retrieve End-of-Text (EoT/EOS) token ID from tokenizer.
    
    Args:
        tokenizer: open_clip tokenizer
        
    Returns:
        int: EoT token ID
        
    Raises:
        AttributeError: If tokenizer doesn't have recognizable EoT token
    """
    if hasattr(tokenizer, 'eot_token_id'):
        return tokenizer.eot_token_id
    elif hasattr(tokenizer, 'eos_token_id'):
        return tokenizer.eos_token_id
    elif hasattr(tokenizer, 'end_token_id'):
        return tokenizer.end_token_id
    else:
        # Fallback for CLIP SimpleTokenizer (classic CLIP BPE)
        tokenizer_type = type(tokenizer).__name__
        if 'SimpleTokenizer' in tokenizer_type or 'CLIPTokenizer' in tokenizer_type:
            return 49407  # Standard CLIP EOT token ID
        raise AttributeError(
            f"Tokenizer {tokenizer_type} does not have "
            "eot_token_id / eos_token_id / end_token_id attribute. "
            "If using custom tokenizer, please add eot_token_id attribute."
        )


def get_pad_token_id(tokenizer) -> int:
    """
    Retrieve padding token ID from tokenizer.
    
    Args:
        tokenizer: open_clip tokenizer
        
    Returns:
        int: Padding token ID (defaults to 0 if not found)
    """
    if hasattr(tokenizer, 'pad_token_id'):
        return tokenizer.pad_token_id
    elif hasattr(tokenizer, 'padding_id'):
        return tokenizer.padding_id
    else:
        # Fallback to 0 (common in CLIP tokenizers)
        return 0


class TextEncoder(nn.Module):
    """
    Extract text features from CLIP text transformer at EoT token position.
    
    Critical: Requires prompts to have fixed context_length to match positional_embedding.
    """
    
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.attn_mask = clip_model.attn_mask
        
        # Context length (typically 77 for CLIP)
        self.context_length = clip_model.context_length
        
        # Validate attn_mask shape matches context_length
        if self.attn_mask is not None and self.attn_mask.shape[0] != self.context_length:
            raise RuntimeError(
                f"attn_mask shape {self.attn_mask.shape} doesn't match "
                f"context_length {self.context_length}"
            )
        
        # Get EoT token ID
        self.eot_token_id = get_eot_token_id(clip_model.tokenizer)
        
    def forward(self, prompts: torch.Tensor, tokenized_prompts: torch.Tensor) -> torch.Tensor:
        """
        Args:
            prompts: [batch_size, context_length, dim] - Embedded prompts
            tokenized_prompts: [batch_size, context_length] - Token IDs for EoT detection
            
        Returns:
            text_features: [batch_size, dim] - Normalized text features at EoT position
        """
        # Validate sequence length matches context_length
        if prompts.size(1) != self.context_length:
            raise ValueError(
                f"Prompt sequence length {prompts.size(1)} doesn't match "
                f"context_length {self.context_length}"
            )
        
        # Use explicit slicing to ensure positional_embedding aligns with context_length
        pos = self.positional_embedding[:self.context_length]
        # Ensure positional embedding is on same device as prompts
        pos = pos.to(prompts.device)
        x = prompts + pos  # [batch_size, context_length, dim]
        
        # Pass to transformer with attn_mask (open_clip expects [batch, seq, dim] format)
        # Ensure attn_mask is on the same device as input
        attn_mask = self.attn_mask
        if attn_mask is not None:
            attn_mask = attn_mask.to(x.device)
        
        x = self.transformer(x, attn_mask=attn_mask)
        x = self.ln_final(x)  # [batch_size, context_length, dim]
        
        # Extract features at EoT position (first occurrence)
        # Ensure tokenized_prompts is on same device
        tokenized_prompts = tokenized_prompts.to(x.device)
        eot_mask = (tokenized_prompts == self.eot_token_id)
        
        # Verify each sequence has at least one EoT token
        if not eot_mask.any(dim=1).all():
            missing_eot = (~eot_mask.any(dim=1)).nonzero(as_tuple=True)[0]
            raise RuntimeError(
                f"EoT token {self.eot_token_id} not found in sequences: {missing_eot.tolist()}"
            )
        
        eot_indices = eot_mask.int().argmax(dim=1).to(torch.int64)
        x = x[torch.arange(x.shape[0], device=x.device), eot_indices]
        
        # Project and normalize
        if self.text_projection is not None:
            x = x @ self.text_projection
        x = x / x.norm(dim=-1, keepdim=True)
        
        return x


class PromptLearner(nn.Module):
    """
    Learnable context vectors for prompt tuning.
    
    Prompt structure (class_token_position='end'):
        [SOS] + [CTX_1] [CTX_2] ... [CTX_n] + [CLASS_TOKEN] [EOS] [PAD]...
    
    Uses "X X X X" placeholder to ensure stable tokenization with fixed context_length.
    
    Args:
        clip_model: open_clip model
        classnames: List of class names (e.g., ['cat', 'dog', ...])
        n_ctx: Number of learnable context tokens (default: 16)
        ctx_init: Optional initialization string (e.g., "a photo of a")
        class_token_position: Only 'end' is supported
        csc: Class-specific context - if True, each class has its own context vectors (default: False)
    """
    
    def __init__(
        self,
        clip_model,
        classnames: List[str],
        n_ctx: int = 16,
        ctx_init: Optional[str] = None,
        class_token_position: str = 'end',
        csc: bool = False,
    ):
        super().__init__()
        
        # Validate class_token_position
        if class_token_position != 'end':
            raise ValueError(
                f"Only class_token_position='end' is supported for stability. "
                f"Got: {class_token_position}"
            )
        
        self.n_cls = len(classnames)
        self.n_ctx = n_ctx
        self.csc = csc
        self.tokenizer = clip_model.tokenizer
        self.context_length = clip_model.context_length
        
        # Get special token IDs (with fallback support)
        sot_token_id = get_sot_token_id(self.tokenizer)
        eot_token_id = get_eot_token_id(self.tokenizer)
        pad_token_id = get_pad_token_id(self.tokenizer)
        
        # Print tokenizer info for debugging (will be logged once during init)
        print(f"[CoOp] Tokenizer: {type(self.tokenizer).__name__}")
        print(f"[CoOp] Special tokens - SOT: {sot_token_id}, EOT: {eot_token_id}, PAD: {pad_token_id}")
        print(f"[CoOp] Context length: {self.context_length}")
        
        # Use token_embedding dtype for text operations
        dtype = clip_model.token_embedding.weight.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        
        # Initialize learnable context vectors
        if ctx_init:
            # Tokenize ctx_init using open_clip tokenizer (returns [1, context_length])
            ctx_init_tokens = self.tokenizer([ctx_init])  # [1, context_length]
            ctx_init_tokens = ctx_init_tokens[0].to(torch.long)  # [context_length], ensure long dtype
            
            # Extract actual context tokens (between SOS and EOS/PAD)
            ctx_token_list = []
            for i, token_id in enumerate(ctx_init_tokens.tolist()):
                if token_id == sot_token_id:
                    continue  # Skip SOS
                elif token_id == eot_token_id or token_id == pad_token_id:
                    break  # Stop at EOS or padding
                else:
                    ctx_token_list.append(token_id)
            
            if len(ctx_token_list) != n_ctx:
                raise ValueError(
                    f"ctx_init token count mismatch: "
                    f"got {len(ctx_token_list)} tokens, expected n_ctx={n_ctx}\n"
                    f"ctx_init: '{ctx_init}'\n"
                    f"tokens: {ctx_token_list}\n"
                    f"Hint: Try adjusting ctx_init wording or n_ctx value"
                )
            
            # Initialize from embedding
            ctx_token_tensor = torch.tensor(ctx_token_list, dtype=torch.long)
            ctx_vectors = clip_model.token_embedding(ctx_token_tensor)  # [n_ctx, dim]
            
            if csc:
                # Expand to all classes: [n_cls, n_ctx, dim]
                ctx_vectors = ctx_vectors.unsqueeze(0).expand(self.n_cls, -1, -1).clone()
            
            prompt_prefix = ctx_init
        else:
            # Random initialization (Gaussian)
            if csc:
                # Class-specific: each class has its own context
                ctx_vectors = torch.empty(self.n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                # Shared: all classes share the same context
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)
        
        # Register as learnable parameter
        # Shape: [n_cls, n_ctx, dim] if CSC, [n_ctx, dim] if shared
        self.ctx = nn.Parameter(ctx_vectors)
        
        # Construct prompts with placeholder (for stable tokenization)
        # Always use "X X X X" placeholder regardless of ctx_init for consistency
        # This ensures token positions are stable across all classes
        placeholder_prefix = " ".join(["X"] * n_ctx)
        classnames_with_prompt = [f"{placeholder_prefix} {name}" for name in classnames]
        
        # Tokenize using open_clip tokenizer (returns [n_cls, context_length])
        tokenized_prompts = self.tokenizer(classnames_with_prompt)
        
        # Ensure long dtype for nn.Embedding compatibility
        tokenized_prompts = tokenized_prompts.to(torch.long)
        
        # Validate shape
        if tokenized_prompts.shape != (self.n_cls, self.context_length):
            raise RuntimeError(
                f"Tokenized prompts shape {tokenized_prompts.shape} doesn't match "
                f"expected [{self.n_cls}, {self.context_length}]"
            )
        
        # Register as buffer for multi-GPU device synchronization
        self.register_buffer("tokenized_prompts", tokenized_prompts)
        
        # Extract class name embeddings ([CLASS_TOKEN] [EOS] [PAD]...)
        with torch.no_grad():
            # Get embeddings for full tokenized prompts
            # Move to same device as token_embedding weights
            device = clip_model.token_embedding.weight.device
            tokenized_prompts_device = tokenized_prompts.to(device)
            full_embeddings = clip_model.token_embedding(tokenized_prompts_device)
            
            # Extract prefix embedding ([SOS])
            self.register_buffer("prefix_embedding", full_embeddings[:, :1, :])
            
            # Extract suffix embeddings (everything after learnable ctx)
            # Structure: [SOS] + [X tokens: 1 to n_ctx] + [CLASS + EOS + PAD: n_ctx+1 to end]
            self.register_buffer("suffix_embedding", full_embeddings[:, 1 + n_ctx:, :])
        
        context_mode = "class-specific" if csc else "shared"
        print(f"[CoOp] PromptLearner initialized: {self.n_cls} classes, {n_ctx} context tokens ({context_mode})")
        
    def forward(self) -> torch.Tensor:
        """
        Construct full prompts by concatenating learnable ctx with fixed embeddings.
        
        Returns:
            prompts: [n_cls, context_length, dim] - Embedded prompts ready for TextEncoder
        """
        ctx = self.ctx
        
        if self.csc:
            # Class-specific context: already [n_cls, n_ctx, dim]
            ctx_expanded = ctx
        else:
            # Shared context: [n_ctx, dim] -> expand to [n_cls, n_ctx, dim]
            ctx_expanded = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        
        # Concatenate: [SOS] + [learnable_ctx] + [class_tokens + EOS + PAD]
        # Ensure all tensors on same device as ctx
        prefix = self.prefix_embedding.to(ctx.device)  # [n_cls, 1, dim]
        suffix = self.suffix_embedding.to(ctx.device)  # [n_cls, context_length - n_ctx - 1, dim]
        
        prompts = torch.cat([prefix, ctx_expanded, suffix], dim=1)
        
        # Validate final shape
        if prompts.shape[1] != self.context_length:
            raise RuntimeError(
                f"Concatenated prompt length {prompts.shape[1]} doesn't match "
                f"context_length {self.context_length}"
            )
        
        return prompts


class CoOpCLIP(nn.Module):
    """
    Complete CoOp model: CLIP with learnable prompts.
    
    Only prompt_learner.ctx is trainable; all other parameters are frozen.
    Uses open_clip's encode_image for stable image feature extraction.
    
    Args:
        clip_model: open_clip model
        classnames: List of class names
        n_ctx: Number of learnable context tokens (default: 16)
        ctx_init: Optional initialization string (e.g., "a photo of a")
        class_token_position: Only 'end' is supported
        csc: Class-specific context - if True, each class has its own context (default: False)
    """
    
    def __init__(
        self,
        clip_model,
        classnames: List[str],
        n_ctx: int = 16,
        ctx_init: Optional[str] = None,
        class_token_position: str = 'end',
        csc: bool = False,
    ):
        super().__init__()
        
        self.prompt_learner = PromptLearner(
            clip_model, classnames, n_ctx, ctx_init, class_token_position, csc
        )
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        
        # Store full clip_model for encode_image (handles projection/normalization)
        self.clip_model = clip_model
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        
        # Global freeze first
        for param in self.parameters():
            param.requires_grad = False
        
        # Then unfreeze only learnable context
        self.prompt_learner.ctx.requires_grad = True
        
        # Set CLIP model to eval mode permanently (frozen encoders)
        self.clip_model.eval()
        
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: [batch_size, 3, H, W]
            
        Returns:
            logits: [batch_size, n_cls] - Classification logits
        """
        # Extract image features using open_clip's encode_image
        # (handles projection/normalization correctly for all backbones)
        with torch.no_grad():
            image_features = self.clip_model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        # Extract text features (with gradients for learnable prompts)
        prompts = self.prompt_learner()
        text_features = self.text_encoder(prompts, self.tokenized_prompts)
        
        # Compute logits (detach logit_scale for clarity)
        logit_scale = self.logit_scale.exp().detach()
        logits = logit_scale * image_features @ text_features.t()
        
        return logits
    
    def get_text_features(self) -> torch.Tensor:
        """
        Extract learned text features for all classes.
        
        Returns:
            text_features: [n_cls, dim] - Normalized text features
        """
        with torch.no_grad():
            prompts = self.prompt_learner()
            text_features = self.text_encoder(prompts, self.tokenized_prompts)
        return text_features
