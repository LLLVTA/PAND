"""
PyTorch Lightning training module for CoOp (Stage A).

Provides sanity checks:
1. Gradient norm monitoring for prompt_learner.ctx
2. Text feature evolution tracking across epochs
"""

import torch
import pytorch_lightning as pl
from typing import Any, Dict, List, Optional
import open_clip

from src.models.components.coop_lightning import CoOpCLIP


class CoOpModule(pl.LightningModule):
    """
    Lightning module for CoOp prompt learning.
    
    Args:
        clip_model_name: open_clip model name (e.g., 'convnext_xxlarge')
        pretrained: open_clip pretrained weights (e.g., 'laion2b_s34b_b82k_augreg_soup')
        classnames: List of class names for prompt learning
        n_ctx: Number of learnable context tokens (default: 16)
        ctx_init: Optional initialization string (e.g., "a photo of a")
        csc: Class-specific context - if True, each class has its own context (default: False)
        learning_rate: Learning rate for SGD (default: 0.002, per CoOp paper)
        momentum: SGD momentum (default: 0.9)
        weight_decay: Weight decay (default: 0.0)
        max_epochs: Max training epochs for cosine scheduler (default: 200)
    """
    
    def __init__(
        self,
        clip_model_name: str,
        pretrained: str,
        classnames: List[str],
        n_ctx: int = 16,
        ctx_init: Optional[str] = None,
        csc: bool = False,
        learning_rate: float = 0.002,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        max_epochs: int = 200,
        use_precomputed_features: bool = False,
    ):
        super().__init__()
        
        # Save hyperparameters
        self.save_hyperparameters(ignore=['classnames'])
        self.use_precomputed_features = use_precomputed_features
        
        # Load open_clip model and transforms
        clip_model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
            clip_model_name, pretrained=pretrained
        )
        
        # Store preprocessing transforms (critical: datamodule must use these!)
        self.preprocess_train = preprocess_train
        self.preprocess_val = preprocess_val
        
        # Attach tokenizer to clip_model (required by CoOpCLIP)
        tokenizer = open_clip.get_tokenizer(clip_model_name)
        
        # Get context_length from model (typically 77 for CLIP)
        if not hasattr(clip_model, 'context_length'):
            # Fallback: infer from positional_embedding shape
            clip_model.context_length = clip_model.positional_embedding.shape[0]
        
        # Validate tokenizer outputs match model's context_length
        test_tokens = tokenizer(["a photo of a dog"])
        if not isinstance(test_tokens, torch.Tensor):
            test_tokens = torch.as_tensor(test_tokens)
        if test_tokens.shape[1] != clip_model.context_length:
            raise RuntimeError(
                f"Tokenizer output length {test_tokens.shape[1]} doesn't match "
                f"model context_length {clip_model.context_length}"
            )
        
        clip_model.tokenizer = tokenizer
        
        # Initialize CoOp model (will freeze encoders internally)
        self.model = CoOpCLIP(
            clip_model=clip_model,
            classnames=classnames,
            n_ctx=n_ctx,
            ctx_init=ctx_init,
            class_token_position='end',
            csc=csc,
        )
        
        # Loss function
        self.criterion = torch.nn.CrossEntropyLoss()
        
        # Store classnames for reference
        self.classnames = classnames
        
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Forward pass through CoOp model.
        
        Args:
            images: Either image tensor [B, C, H, W] or precomputed features [B, D]
        """
        if self.use_precomputed_features:
            # images is actually precomputed features [B, D], already normalized
            image_features = images
            
            # Get text features with gradients (for prompt learning)
            prompts = self.model.prompt_learner()
            text_features = self.model.text_encoder(prompts, self.model.tokenized_prompts)
            
            # Compute logits manually (same as CoOpCLIP.forward)
            logit_scale = self.model.logit_scale.exp().detach()
            logits = logit_scale * image_features @ text_features.t()
            return logits
        else:
            # Standard forward with image encoding
            return self.model(images)
    
    def training_step(self, batch, batch_idx) -> torch.Tensor:
        images, labels = batch
        logits = self(images)
        loss = self.criterion(logits, labels)
        
        # Log training metrics
        acc = (logits.argmax(dim=1) == labels).float().mean()
        self.log('train/loss', loss, prog_bar=True)
        self.log('train/acc', acc, prog_bar=True)
        
        # Log learning rate for monitoring
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('train/lr', current_lr, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        loss = self.criterion(logits, labels)
        acc = (logits.argmax(dim=1) == labels).float().mean()
        
        # sync_dist=True to aggregate metrics across DDP ranks so checkpoint monitor sees global val_acc
        self.log('val/loss', loss, prog_bar=True, sync_dist=True)
        self.log('val/acc', acc, prog_bar=True, sync_dist=True)
        self.log('val_acc', acc, prog_bar=False, sync_dist=True)  # alias without slash for checkpoint monitoring
        
    def on_after_backward(self):
        """Sanity check: Monitor gradient norm for learnable context (rank 0 only)."""
        # Guard against trainer not being bound or non-rank0
        if not getattr(self, "trainer", None) or not self.trainer.is_global_zero:
            return
        
        if self.model.prompt_learner.ctx.grad is not None:
            grad_norm = self.model.prompt_learner.ctx.grad.norm().item()
            self.log('sanity/ctx_grad_norm', grad_norm)
        
    def on_train_epoch_end(self):
        """Sanity check: Track text feature evolution across epochs (rank 0 only, fp32)."""
        # Guard against trainer not being bound or non-rank0
        if not getattr(self, "trainer", None) or not self.trainer.is_global_zero:
            return
        
        # Compute in fp32 to avoid AMP numerical noise (CPU/CUDA compatible)
        with torch.no_grad():
            # Disable autocast based on device type
            device_type = self.device.type
            if device_type == "cuda":
                autocast_ctx = torch.cuda.amp.autocast(enabled=False)
            else:
                # CPU or other devices: use identity context
                from contextlib import nullcontext
                autocast_ctx = nullcontext()
            
            with autocast_ctx:
                text_features = self.model.get_text_features().float()  # [n_cls, dim]
                
                # Compute mean norm and pairwise similarity
                mean_norm = text_features.norm(dim=-1).mean().item()
                similarity_matrix = text_features @ text_features.t()  # [n_cls, n_cls]
                
                # Average off-diagonal similarity
                n_cls = len(self.classnames)
                off_diag_mask = ~torch.eye(n_cls, dtype=torch.bool, device=similarity_matrix.device)
                mean_similarity = similarity_matrix[off_diag_mask].mean().item()
                
                self.log('sanity/text_feat_norm', mean_norm)
                self.log('sanity/text_feat_similarity', mean_similarity)
    
    def configure_optimizers(self):
        """
        Configure SGD optimizer with cosine annealing scheduler (per CoOp paper).
        """
        # Only optimize prompt_learner.ctx
        optimizer = torch.optim.SGD(
            [self.model.prompt_learner.ctx],
            lr=self.hparams.learning_rate,
            momentum=self.hparams.momentum,
            weight_decay=self.hparams.weight_decay
        )
        
        # Cosine annealing scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.max_epochs,
            eta_min=0.0
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1
            }
        }
    
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """
        Override to avoid saving large text_features in checkpoint.
        Only save minimal metadata needed for feature extraction.
        """
        # Remove optimizer state to reduce checkpoint size (optional)
        # checkpoint.pop('optimizer_states', None)
        
        # Add classnames for feature extraction script
        checkpoint['classnames'] = self.classnames
