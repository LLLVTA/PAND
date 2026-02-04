"""
Extract learned text features from CoOp checkpoint for VL2Lite Stage B.

Usage:
    python scripts/extract_coop_features.py \
        --checkpoint path/to/coop/checkpoint.ckpt \
        --output path/to/coop_text_features.pt \
        --clip_model_name convnext_xxlarge \
        --pretrained laion2b_s34b_b82k_augreg_soup

Output format:
    {
        'text_features': torch.Tensor,  # [n_cls, dim], normalized, fp32
        'classnames': List[str],
        'n_ctx': int,
        'clip_model_name': str,
        'pretrained': str
    }
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import argparse
import torch
import open_clip

from src.models.components.coop_lightning import CoOpCLIP


def extract_features(
    checkpoint_path: str,
    output_path: str,
    clip_model_name: str,
    pretrained: str,
    device: str = "cuda"
):
    """
    Extract text features from CoOp checkpoint with strict validation.
    
    Args:
        checkpoint_path: Path to Lightning checkpoint (.ckpt)
        output_path: Output path for extracted features (.pt)
        clip_model_name: open_clip model name
        pretrained: open_clip pretrained weights name
        device: Device for feature extraction ('cuda' or 'cpu')
    """
    print(f"Loading checkpoint from: {checkpoint_path}")
    print(f"Using device: {device}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Validate checkpoint structure
    if 'state_dict' not in checkpoint:
        raise ValueError(
            "Checkpoint missing 'state_dict'. "
            f"Available keys: {list(checkpoint.keys())}"
        )
    
    # Extract metadata
    if 'classnames' not in checkpoint:
        raise ValueError("Checkpoint missing 'classnames' metadata")
    
    classnames = checkpoint['classnames']
    hparams = checkpoint.get('hyper_parameters', {})
    n_ctx = hparams.get('n_ctx', 16)
    ctx_init = hparams.get('ctx_init', None)
    
    print(f"Classes: {len(classnames)}")
    print(f"n_ctx: {n_ctx}")
    print(f"ctx_init: {ctx_init}")
    
    # Load CLIP model
    print(f"Loading CLIP model: {clip_model_name} ({pretrained})")
    clip_model, _, _ = open_clip.create_model_and_transforms(
        clip_model_name, pretrained=pretrained
    )
    clip_model = clip_model.to(device)
    clip_model.eval()
    
    # Attach tokenizer to clip_model (critical: must match training setup)
    tokenizer = open_clip.get_tokenizer(clip_model_name)
    
    # Get context_length from model (typically 77 for CLIP)
    if not hasattr(clip_model, 'context_length'):
        # Fallback: infer from positional_embedding shape
        clip_model.context_length = clip_model.positional_embedding.shape[0]
        print(f"Inferred context_length from positional_embedding: {clip_model.context_length}")
    
    # Validate tokenizer outputs match model's context_length
    test_tokens = tokenizer(["a photo of a dog"])
    if not isinstance(test_tokens, torch.Tensor):
        test_tokens = torch.as_tensor(test_tokens)
    if test_tokens.shape[1] != clip_model.context_length:
        raise RuntimeError(
            f"Tokenizer output length {test_tokens.shape[1]} doesn't match "
            f"model context_length {clip_model.context_length}. "
            "This may cause shape mismatch during feature extraction."
        )
    
    clip_model.tokenizer = tokenizer
    print(f"✓ Tokenizer validated: context_length={clip_model.context_length}")
    
    # Initialize CoOp model
    # Note: Use ctx_init=None because we'll restore ctx from checkpoint.
    # This avoids strict token count validation that may fail due to tokenizer differences.
    print(f"Initializing CoOp model (ctx_init will be restored from checkpoint)...")
    coop_model = CoOpCLIP(
        clip_model=clip_model,
        classnames=classnames,
        n_ctx=n_ctx,
        ctx_init=None,  # Will be overwritten by checkpoint
        class_token_position='end'
    )
    
    # Find and restore only the ctx parameter (most robust approach)
    # Buffers (tokenized_prompts, prefix/suffix_embedding) will be regenerated
    # based on current environment's tokenizer, ensuring consistency.
    state_dict = checkpoint['state_dict']
    
    # More strict matching: must end with 'prompt_learner.ctx' AND contain '.prompt_learner.ctx'
    # This avoids false matches from unrelated modules
    ctx_candidates = [
        key for key in state_dict.keys()
        if key.endswith('prompt_learner.ctx') and '.prompt_learner.ctx' in key
    ]
    
    if len(ctx_candidates) == 0:
        raise ValueError(
            "Cannot find 'prompt_learner.ctx' in state_dict. "
            f"Available keys: {list(state_dict.keys())[:10]}"
        )
    elif len(ctx_candidates) > 1:
        raise ValueError(
            f"Found multiple ctx candidates: {ctx_candidates}. "
            "Cannot determine which one to use. Please check checkpoint structure."
        )
    
    ctx_key = ctx_candidates[0]
    print(f"Found ctx parameter: '{ctx_key}'")
    
    # Restore ctx parameter
    ctx_param = state_dict[ctx_key]
    
    # Ensure dtype matches (in case checkpoint was saved with AMP fp16/bf16)
    target_dtype = coop_model.prompt_learner.ctx.dtype
    if ctx_param.dtype != target_dtype:
        print(f"Converting ctx dtype from {ctx_param.dtype} to {target_dtype}")
        ctx_param = ctx_param.to(dtype=target_dtype)
    
    if ctx_param.shape != coop_model.prompt_learner.ctx.shape:
        raise RuntimeError(
            f"ctx shape mismatch: checkpoint has {ctx_param.shape}, "
            f"but model expects {coop_model.prompt_learner.ctx.shape}. "
            f"Check n_ctx={n_ctx} matches the checkpoint."
        )
    
    coop_model.prompt_learner.ctx.data.copy_(ctx_param)
    print(f"✓ Restored ctx parameter: shape={ctx_param.shape}, dtype={ctx_param.dtype}")
    
    # Move model to device
    coop_model = coop_model.to(device)
    coop_model.eval()
    
    # Extract text features
    print("Extracting text features...")
    with torch.no_grad(), torch.autocast(device_type=device, enabled=False):
        text_features = coop_model.get_text_features()  # [n_cls, dim]
    
    # Convert to fp32 for compatibility
    text_features = text_features.float().cpu()
    
    # Verify normalization
    norms = text_features.norm(dim=-1)
    mean_norm = norms.mean().item()
    if not torch.allclose(norms, torch.ones_like(norms), atol=1e-5):
        print(f"⚠ Warning: Features not perfectly normalized (mean norm: {mean_norm:.6f})")
    else:
        print(f"✓ Features normalized (mean norm: {mean_norm:.6f})")
    
    # Save output
    output_data = {
        'text_features': text_features,
        'classnames': classnames,
        'n_ctx': n_ctx,
        'ctx_init': ctx_init,  # Save for reference (but not used in restoration)
        'clip_model_name': clip_model_name,
        'pretrained': pretrained,
        'context_length': clip_model.context_length,
    }
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output_data, output_path)
    
    print(f"\n{'='*60}")
    print("✓ Text features extracted successfully!")
    print(f"{'='*60}")
    print(f"Output file: {output_path}")
    print(f"Shape: {text_features.shape}")
    print(f"Dtype: {text_features.dtype}")
    print(f"Mean norm: {mean_norm:.6f}")
    print(f"Classes: {len(classnames)}")
    print(f"Context tokens: {n_ctx}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Extract CoOp text features")
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to CoOp checkpoint (.ckpt)')
    parser.add_argument('--output', type=str, required=True,
                        help='Output path for text features (.pt)')
    parser.add_argument('--clip_model_name', type=str, default='convnext_xxlarge',
                        help='open_clip model name')
    parser.add_argument('--pretrained', type=str, default='laion2b_s34b_b82k_augreg_soup',
                        help='open_clip pretrained weights')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device for feature extraction (default: cuda)')
    
    args = parser.parse_args()
    
    # Check CUDA availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU")
        args.device = 'cpu'
    
    extract_features(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        clip_model_name=args.clip_model_name,
        pretrained=args.pretrained,
        device=args.device
    )


if __name__ == '__main__':
    main()
