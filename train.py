import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
import argparse

from utils.dataset import VCTKDataset
from src.model.vae import VAE

# --- Load selected speakers ---
SELECTED_SPEAKERS_PATH = "src/speaker_embedding/selected_speakers.txt"
with open(SELECTED_SPEAKERS_PATH, "r") as f:
    selected_speakers = f.read().replace("'", "").replace("\n", "").split(",")
    selected_speakers = [s.strip() for s in selected_speakers if s.strip()]


# --- Config ---
def get_device():
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        print("Using MPS (Apple Silicon GPU)")
        return torch.device("mps")
    elif torch.cuda.is_available():
        print("Using CUDA GPU")
        return torch.device("cuda")
    else:
        print("Using CPU")
        return torch.device("cpu")


parser = argparse.ArgumentParser()
parser.add_argument(
    "--device",
    type=str,
    default=None,
    help="Device to use: cpu, cuda, or mps (default: auto)",
)
parser.add_argument(
    "--use_tacotron2",
    action="store_true",
    default=True,
    help="Use Tacotron2 as decoder (default: True)",
)
parser.add_argument(
    "--fine_tune_decoder",
    action="store_true",
    default=True,
    help="Fine-tune Tacotron2 decoder (default: True)",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=4,
    help="Batch size (reduced for Tacotron2 memory usage)",
)
parser.add_argument(
    "--epochs",
    type=int,
    default=20,
    help="Number of training epochs",
)
parser.add_argument(
    "--vae_lr",
    type=float,
    default=1e-4,
    help="Learning rate for VAE components",
)
parser.add_argument(
    "--decoder_lr",
    type=float,
    default=1e-5,
    help="Learning rate for Tacotron2 decoder",
)
parser.add_argument(
    "--postnet_lr",
    type=float,
    default=1e-5,
    help="Learning rate for Tacotron2 postnet",
)
parser.add_argument(
    "--resume",
    type=str,
    default=None,
    help="Path to checkpoint to resume training from",
)
args = parser.parse_args()

if args.device:
    DEVICE = torch.device(args.device)
    print(f"Using device: {DEVICE}")
else:
    DEVICE = get_device()

DATA_ROOT = "dataset/VCTK-Corpus-0"
SPEAKER_EMB_PATH = "src/speaker_embedding/speaker_embeddings.npy"
BATCH_SIZE = args.batch_size
EPOCHS = args.epochs
CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# --- Load speaker embeddings ---
speaker_embeddings = np.load(
    SPEAKER_EMB_PATH, allow_pickle=True
).item()  # dict: spk_id -> emb (256,)

# --- Dataset and DataLoader ---
dataset = VCTKDataset(DATA_ROOT, cache_mel=True)
# Filter dataset to only include selected speakers
selected_indices = [
    i
    for i, item in enumerate(dataset.items)
    if any(spk in str(item[0]) for spk in selected_speakers)
]
dataset.items = [dataset.items[i] for i in selected_indices]


def collate_fn(batch):
    # Use the dataset's collate_cvae, but also return speaker ids as list of strings
    text_pad, text_lens, mel_pad, mel_lens, gate_pad, spk_ids = (
        VCTKDataset.collate_cvae(batch)
    )
    return text_pad, text_lens, mel_pad, mel_lens, gate_pad, spk_ids


dataloader = DataLoader(
    dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
)

# --- Models ---
# Initialize VAE with Tacotron2 decoder and fine-tuning
vae = VAE(
    n_mels=80,
    spk_emb_dim=256,
    latent_dim=64,
    hidden_dim=256,
    use_tacotron2=args.use_tacotron2,
    fine_tune_decoder=args.fine_tune_decoder,
    device=DEVICE,
).to(DEVICE)

# Print model information
print("\n=== Model Information ===")
total_params, trainable_params = vae.count_parameters()
print(f"Model initialized with fine_tune_decoder={args.fine_tune_decoder}")

# --- Optimizer & Scheduler ---
if args.use_tacotron2 and args.fine_tune_decoder:
    # Use parameter groups with different learning rates
    param_groups = vae.get_parameter_groups(
        vae_lr=args.vae_lr, decoder_lr=args.decoder_lr, postnet_lr=args.postnet_lr
    )
    optimizer = optim.Adam(param_groups)
    print("\n=== Parameter Groups ===")
    for group in param_groups:
        print(
            f"{group['name']}: lr={group['lr']}, params={sum(p.numel() for p in group['params']):,}"
        )
else:
    # Use standard optimizer for all parameters
    optimizer = optim.Adam(vae.parameters(), lr=args.vae_lr)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, "min", patience=5, factor=0.5, verbose=True
)

# --- Load checkpoint if resuming ---
start_epoch = 1
if args.resume and os.path.exists(args.resume):
    print(f"\n=== Loading checkpoint: {args.resume} ===")
    checkpoint = torch.load(args.resume, map_location=DEVICE)

    # Load model state
    vae.load_state_dict(checkpoint["vae"])

    # Load optimizer and scheduler state
    if args.use_tacotron2 and args.fine_tune_decoder:
        # Recreate parameter groups and optimizer
        param_groups = vae.get_parameter_groups(
            vae_lr=args.vae_lr, decoder_lr=args.decoder_lr, postnet_lr=args.postnet_lr
        )
        optimizer = optim.Adam(param_groups)
    else:
        optimizer = optim.Adam(vae.parameters(), lr=args.vae_lr)

    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])

    # Get starting epoch
    start_epoch = checkpoint["epoch"] + 1
    best_loss = checkpoint["loss"]

    print(f"Resuming from epoch {start_epoch}")
    print(f"Previous best loss: {best_loss:.4f}")

    # Verify configuration matches
    if checkpoint.get("fine_tune_decoder") != args.fine_tune_decoder:
        print("Warning: fine_tune_decoder setting differs from checkpoint")
    if checkpoint.get("use_tacotron2") != args.use_tacotron2:
        print("Warning: use_tacotron2 setting differs from checkpoint")
else:
    best_loss = float("inf")


# --- KL Annealing ---
def get_kl_weight(epoch, kl_anneal_epochs=10, max_kl_weight=1e-4):
    return min(max_kl_weight, max_kl_weight * epoch / kl_anneal_epochs)


# --- Loss ---
def vae_loss(recon, target, mu, logvar, kl_weight):
    recon_loss = F.mse_loss(recon, target)  # Changed to MSE for better stability
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_weight * kld, recon_loss, kld


# --- Validation ---
def validate(vae, dataloader, device, speaker_embeddings, dataset, use_tacotron2=True):
    """Validate the model on a subset of data."""
    vae.eval()
    total_loss, total_recon, total_kld = 0, 0, 0
    num_batches = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= 10:  # Only validate on first 10 batches
                break

            text_pad, text_lens, mel_pad, mel_lens, gate_pad, spk_ids = batch
            mel_pad = mel_pad.to(device)

            # Extract speaker embeddings
            spk_emb_batch = []
            for i, spk_id in enumerate(spk_ids):
                for k, v in dataset.spk2id.items():
                    if v == spk_id.item():
                        spk_str = k
                        break
                spk_emb = speaker_embeddings[spk_str]
                spk_emb_batch.append(spk_emb)
            spk_emb_batch = torch.tensor(
                np.stack(spk_emb_batch), dtype=torch.float32, device=device
            )

            # Prepare text for Tacotron2
            if use_tacotron2:
                batch_texts = []
                for i in range(len(spk_ids)):
                    item_idx = batch_idx * BATCH_SIZE + i
                    if item_idx < len(dataset.items):
                        text = extract_text_from_dataset_item(dataset.items[item_idx])
                    else:
                        text = "Hello world"
                    batch_texts.append(text)
            else:
                batch_texts = None

            try:
                if use_tacotron2 and batch_texts:
                    recon, mu, logvar = vae(mel_pad, spk_emb_batch, batch_texts)
                else:
                    # If Tacotron2 is enabled but no text, skip this batch
                    if use_tacotron2:
                        print(
                            f"Skipping validation batch {batch_idx}: Tacotron2 enabled but no text available"
                        )
                        continue
                    else:
                        # Use fallback decoder only when Tacotron2 is disabled
                        recon, mu, logvar = vae(mel_pad, spk_emb_batch, None)

                loss, recon_loss, kld = vae_loss(
                    recon, mel_pad, mu, logvar, kl_weight=1e-4
                )

                total_loss += loss.item()
                total_recon += recon_loss.item()
                total_kld += kld.item()
                num_batches += 1

            except Exception as e:
                print(f"Error in validation batch {batch_idx}: {e}")
                continue

    if num_batches > 0:
        avg_loss = total_loss / num_batches
        avg_recon = total_recon / num_batches
        avg_kld = total_kld / num_batches
        return avg_loss, avg_recon, avg_kld
    else:
        return float("inf"), float("inf"), float("inf")


# --- Text preprocessing for Tacotron2 ---
def extract_text_from_dataset_item(item):
    """Extract text from dataset item for Tacotron2 input."""
    # This is a simplified version - you might need to adjust based on your dataset structure
    if isinstance(item, (list, tuple)) and len(item) > 1:
        # Assuming item[1] contains the text
        return str(item[1])
    else:
        # Fallback to a default text
        return "Hello world"


best_loss = float("inf")

# --- Training Loop ---
print(f"\n=== Starting Training ===")
print(f"Epochs: {EPOCHS}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Use Tacotron2: {args.use_tacotron2}")
print(f"Fine-tune decoder: {args.fine_tune_decoder}")
print(f"Starting from epoch: {start_epoch}")

for epoch in range(start_epoch, EPOCHS + 1):
    vae.train()
    total_loss, total_recon, total_kld = 0, 0, 0
    kl_weight = get_kl_weight(epoch)
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch_idx, batch in enumerate(pbar):
        text_pad, text_lens, mel_pad, mel_lens, gate_pad, spk_ids = batch
        mel_pad = mel_pad.to(DEVICE)

        # Extract speaker embeddings
        spk_emb_batch = []
        for i, spk_id in enumerate(spk_ids):
            for k, v in dataset.spk2id.items():
                if v == spk_id.item():
                    spk_str = k
                    break
            spk_emb = speaker_embeddings[spk_str]
            spk_emb_batch.append(spk_emb)
        spk_emb_batch = torch.tensor(
            np.stack(spk_emb_batch), dtype=torch.float32, device=DEVICE
        )

        # Prepare text for Tacotron2
        if args.use_tacotron2:
            # Extract text from dataset items
            batch_texts = []
            for i in range(len(spk_ids)):
                # Get the corresponding dataset item
                item_idx = batch_idx * BATCH_SIZE + i
                if item_idx < len(dataset.items):
                    text = extract_text_from_dataset_item(dataset.items[item_idx])
                else:
                    text = "Hello world"  # Fallback
                batch_texts.append(text)
        else:
            batch_texts = None

        # Forward VAE
        try:
            if args.use_tacotron2 and batch_texts:
                recon, mu, logvar = vae(mel_pad, spk_emb_batch, batch_texts)
            else:
                # If Tacotron2 is enabled but no text, skip this batch
                if args.use_tacotron2:
                    print(
                        f"Skipping batch {batch_idx}: Tacotron2 enabled but no text available"
                    )
                    continue
                else:
                    # Use fallback decoder only when Tacotron2 is disabled
                    recon, mu, logvar = vae(mel_pad, spk_emb_batch, None)

            loss, recon_loss, kld = vae_loss(recon, mel_pad, mu, logvar, kl_weight)

            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if args.use_tacotron2 and args.fine_tune_decoder:
                # Clip gradients for each parameter group separately
                for group in param_groups:
                    torch.nn.utils.clip_grad_norm_(group["params"], max_norm=1.0)
            else:
                torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kld += kld.item()

            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.3f}",
                    "recon": f"{recon_loss.item():.3f}",
                    "kld": f"{kld.item():.3f}",
                    "klw": f"{kl_weight:.1e}",
                }
            )

        except Exception as e:
            print(f"Error in batch {batch_idx}: {e}")
            continue

    avg_loss = total_loss / len(dataloader)
    avg_recon = total_recon / len(dataloader)
    avg_kld = total_kld / len(dataloader)

    print(
        f"Epoch {epoch}: loss={avg_loss:.4f}, recon={avg_recon:.4f}, kld={avg_kld:.4f}"
    )

    # Validation
    if epoch % 5 == 0:
        val_loss, val_recon, val_kld = validate(
            vae, dataloader, DEVICE, speaker_embeddings, dataset, args.use_tacotron2
        )
        print(
            f"Validation: loss={val_loss:.4f}, recon={val_recon:.4f}, kld={val_kld:.4f}"
        )

    scheduler.step(avg_loss)

    # Save checkpoint every 5 epochs
    if epoch % 5 == 0:
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"vae_tacotron2_epoch{epoch}.pt")
        torch.save(
            {
                "epoch": epoch,
                "vae": vae.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "loss": avg_loss,
                "use_tacotron2": args.use_tacotron2,
                "fine_tune_decoder": args.fine_tune_decoder,
                "args": vars(args),
            },
            checkpoint_path,
        )
        print(f"Checkpoint saved: {checkpoint_path}")

    # Save best model
    if avg_loss < best_loss:
        best_loss = avg_loss
        best_model_path = os.path.join(CHECKPOINT_DIR, "vae_tacotron2_best.pt")
        torch.save(
            {
                "epoch": epoch,
                "vae": vae.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "loss": best_loss,
                "use_tacotron2": args.use_tacotron2,
                "fine_tune_decoder": args.fine_tune_decoder,
                "args": vars(args),
            },
            best_model_path,
        )
        print(f"New best model saved: {best_model_path}")

print("\n=== Training Completed ===")
print(f"Best loss: {best_loss:.4f}")
print(f"Best model saved to: {os.path.join(CHECKPOINT_DIR, 'vae_tacotron2_best.pt')}")
