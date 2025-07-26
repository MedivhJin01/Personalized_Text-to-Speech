import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    """
    Variational Autoencoder for TTS using Tacotron2 as decoder.
    - Encoder: takes mel spectrogram and speaker embedding to encode audio and speaker identity.
    - Latent: mean, logvar, reparameterization for speaker identity.
    - Decoder: Uses Tacotron2 to generate mel from text and latent speaker representation.
    """

    def __init__(
        self,
        n_mels=80,
        spk_emb_dim=256,
        latent_dim=64,
        hidden_dim=256,
        use_tacotron2=True,
        device="cuda",
    ):
        super().__init__()
        self.n_mels = n_mels
        self.spk_emb_dim = spk_emb_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.use_tacotron2 = use_tacotron2
        self.device = device

        # Encoder: input = [mel, speaker_emb]
        self.encoder_prenet = nn.Sequential(
            nn.Conv1d(n_mels + spk_emb_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )
        self.encoder_lstm = nn.LSTM(
            hidden_dim, hidden_dim, batch_first=True, bidirectional=True
        )
        self.fc_mu = nn.Linear(hidden_dim * 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim * 2, latent_dim)

        # Tacotron2 decoder
        if use_tacotron2:
            try:
                import torch.hub

                self.tacotron2 = torch.hub.load(
                    "NVIDIA/DeepLearningExamples:torchhub",
                    "nvidia_tacotron2",
                    model_math="fp16",
                )
                self.tacotron2 = self.tacotron2.to(device)
                self.tacotron2.eval()

                # Load utilities for text preprocessing
                self.tts_utils = torch.hub.load(
                    "NVIDIA/DeepLearningExamples:torchhub", "nvidia_tts_utils"
                )
                print("Tacotron2 loaded successfully")
            except Exception as e:
                print(f"Failed to load Tacotron2: {e}")
                self.use_tacotron2 = False

        # Fallback decoder if Tacotron2 is not available
        if not use_tacotron2:
            self.decoder_input_proj = nn.Linear(latent_dim, hidden_dim)
            self.decoder_lstm = nn.LSTM(
                hidden_dim, hidden_dim, batch_first=True, bidirectional=True
            )
            self.decoder_out = nn.Sequential(
                nn.Linear(hidden_dim * 2, n_mels),
                nn.Tanh(),
            )

    def encode(self, mel, spk_emb):
        """
        Encode mel spectrogram and speaker embedding into latent space.
        mel: [B, n_mels, T]
        spk_emb: [B, spk_emb_dim]
        """
        B, n_mels, T_mel = mel.shape

        # Expand speaker embedding to [B, spk_emb_dim, T_mel]
        spk_emb_exp = spk_emb.unsqueeze(-1).expand(-1, -1, T_mel)

        # Concatenate along channel dim: [B, n_mels + spk_emb_dim, T_mel]
        x = torch.cat([mel, spk_emb_exp], dim=1)
        x = self.encoder_prenet(x)

        # [B, hidden_dim, T_mel] -> [B, T_mel, hidden_dim]
        x = x.transpose(1, 2)
        x, _ = self.encoder_lstm(x)

        # Take mean over time for global representation
        x_mean = x.mean(dim=1)
        mu = self.fc_mu(x_mean)
        logvar = self.fc_logvar(x_mean)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode_with_tacotron2(self, z, text):
        """
        Use Tacotron2 to generate mel spectrogram from text and latent speaker representation.
        z: [B, latent_dim] - latent speaker representation
        text: list of strings
        """
        if not self.use_tacotron2:
            raise RuntimeError("Tacotron2 not available")

        # Prepare text input for Tacotron2
        sequences, lengths = self.tts_utils.prepare_input_sequence(text)
        sequences = sequences.to(self.device)
        lengths = lengths.to(self.device)

        # Modify Tacotron2 to use our latent speaker representation
        # We'll inject the latent z into the Tacotron2's speaker embedding
        with torch.no_grad():
            # Store original speaker embedding if it exists
            original_speaker_embedding = None
            if hasattr(self.tacotron2, "speaker_embedding"):
                original_speaker_embedding = self.tacotron2.speaker_embedding

            # Replace speaker embedding with our latent representation
            # Project latent to speaker embedding dimension if needed
            if hasattr(self.tacotron2, "speaker_embedding"):
                spk_proj = nn.Linear(
                    self.latent_dim, self.tacotron2.speaker_embedding.weight.size(1)
                ).to(self.device)
                projected_z = spk_proj(z)
                self.tacotron2.speaker_embedding.weight.data = projected_z.unsqueeze(
                    0
                ).expand_as(self.tacotron2.speaker_embedding.weight)

            # Generate mel spectrogram
            mel, _, _ = self.tacotron2.infer(sequences, lengths)

            # Restore original speaker embedding if it existed
            if original_speaker_embedding is not None:
                self.tacotron2.speaker_embedding.weight.data = (
                    original_speaker_embedding.weight.data
                )

        return mel

    def decode_fallback(self, z, T):
        """
        Fallback decoder when Tacotron2 is not available.
        z: [B, latent_dim]
        T: int, target length
        """
        B = z.size(0)
        # Expand z to all time steps
        z_exp = z.unsqueeze(1).expand(-1, T, -1)
        dec_in = self.decoder_input_proj(z_exp)
        x, _ = self.decoder_lstm(dec_in)
        out = self.decoder_out(x)
        # [B, T, n_mels] -> [B, n_mels, T]
        return out.transpose(1, 2)

    def forward(self, mel, spk_emb, text=None):
        """
        Forward pass: encode mel and speaker embedding, then decode with Tacotron2.
        mel: [B, n_mels, T]
        spk_emb: [B, spk_emb_dim]
        text: list of strings (required if using Tacotron2)
        """
        mu, logvar = self.encode(mel, spk_emb)
        z = self.reparameterize(mu, logvar)

        if self.use_tacotron2 and text is not None:
            # Use Tacotron2 for decoding
            recon = self.decode_with_tacotron2(z, text)
        else:
            # Use fallback decoder
            T = mel.size(2)
            recon = self.decode_fallback(z, T)

        return recon, mu, logvar

    def synthesize(self, text, spk_emb, z=None):
        """
        Synthesize speech from text and speaker embedding or latent representation.
        text: list of strings
        spk_emb: [B, spk_emb_dim] or None
        z: [B, latent_dim] or None (if provided, spk_emb is ignored)
        """
        if not self.use_tacotron2:
            raise RuntimeError("Tacotron2 not available for synthesis")

        if z is None and spk_emb is None:
            raise ValueError("Either spk_emb or z must be provided")

        if z is None:
            # Use speaker embedding directly (no encoding needed)
            # Project speaker embedding to latent space for consistency
            z = spk_emb  # For now, assume spk_emb_dim == latent_dim

        return self.decode_with_tacotron2(z, text)
