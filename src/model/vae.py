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
        fine_tune_decoder=True,
        device="cuda",
    ):
        super().__init__()
        self.n_mels = n_mels
        self.spk_emb_dim = spk_emb_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.use_tacotron2 = use_tacotron2
        self.fine_tune_decoder = fine_tune_decoder
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

        # Speaker conditioning projection (for latent space to hidden space)
        self.speaker_projection = nn.Linear(latent_dim, hidden_dim)
        # Direct projection from speaker embedding to decoder dimension (for inference)
        self.speaker_to_decoder = nn.Linear(spk_emb_dim, hidden_dim)

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

                # Configure fine-tuning settings
                if fine_tune_decoder:
                    # Enable gradients for decoder components
                    self.tacotron2.decoder.requires_grad_(True)
                    self.tacotron2.postnet.requires_grad_(True)
                    # Try to access attention through decoder, fallback to direct access
                    if hasattr(self.tacotron2.decoder, "attention"):
                        self.tacotron2.decoder.attention.requires_grad_(True)
                    elif hasattr(self.tacotron2, "attention"):
                        self.tacotron2.attention.requires_grad_(True)
                    # Keep encoder frozen for stability
                    self.tacotron2.encoder.requires_grad_(False)
                    self.tacotron2.embedding.requires_grad_(False)
                else:
                    # Freeze all Tacotron2 components
                    self.tacotron2.requires_grad_(False)

                self.tacotron2.eval()

                # Load utilities for text preprocessing
                self.tts_utils = torch.hub.load(
                    "NVIDIA/DeepLearningExamples:torchhub", "nvidia_tts_utils"
                )
                print(
                    f"Tacotron2 loaded successfully (fine_tune_decoder={fine_tune_decoder})"
                )
            except Exception as e:
                print(f"Failed to load Tacotron2: {e}")
                print("Falling back to custom decoder...")
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
        Use Tacotron2's standard forward pass with speaker conditioning.
        z: [B, latent_dim] - latent speaker representation
        text: list of strings
        """
        if not self.use_tacotron2:
            raise RuntimeError("Tacotron2 not available")

        # Prepare text input for Tacotron2
        sequences, lengths = self.tts_utils.prepare_input_sequence(text)
        sequences = sequences.to(self.device)
        lengths = lengths.to(self.device)

        # Use no_grad only if not fine-tuning
        if not self.fine_tune_decoder:
            context = torch.no_grad()
        else:
            context = torch.enable_grad()

        with context:
            # Use Tacotron2's standard forward pass
            try:
                # Try to get 4 values from inference
                mel_outputs, mel_outputs_postnet, _, alignments = self.tacotron2.infer(
                    sequences, lengths
                )
            except ValueError:
                # If that fails, try to get 3 values
                try:
                    mel_outputs, mel_outputs_postnet, alignments = self.tacotron2.infer(
                        sequences, lengths
                    )
                except ValueError:
                    # If that also fails, try to get 2 values
                    mel_outputs, mel_outputs_postnet = self.tacotron2.infer(
                        sequences, lengths
                    )
                    # Create dummy alignments
                    alignments = torch.zeros(
                        sequences.size(0), 1, sequences.size(1)
                    ).to(self.device)

            # Apply speaker conditioning to the mel outputs
            # Project latent speaker representation to mel dimension
            speaker_conditioning = self.speaker_projection(z)  # [B, hidden_dim]

            # Project to mel dimension for conditioning
            if speaker_conditioning.size(-1) != self.n_mels:
                spk_to_mel = nn.Linear(speaker_conditioning.size(-1), self.n_mels).to(
                    self.device
                )
                speaker_conditioning = spk_to_mel(speaker_conditioning)

            # Add speaker conditioning to mel outputs
            # Expand speaker conditioning to match mel output dimensions
            if mel_outputs_postnet.dim() == 3:  # [B, n_mels, T]
                speaker_conditioning = speaker_conditioning.unsqueeze(-1).expand(
                    -1, -1, mel_outputs_postnet.size(-1)
                )
            else:  # [B, T, n_mels]
                speaker_conditioning = speaker_conditioning.unsqueeze(1).expand(
                    -1, mel_outputs_postnet.size(1), -1
                )

            mel_outputs_conditioned = mel_outputs_postnet + speaker_conditioning

        return mel_outputs_conditioned

    def enable_fine_tuning(self):
        """Enable fine-tuning of Tacotron2 decoder components."""
        if self.use_tacotron2:
            self.fine_tune_decoder = True
            self.tacotron2.decoder.requires_grad_(True)
            self.tacotron2.postnet.requires_grad_(True)
            # Try to access attention through decoder, fallback to direct access
            if hasattr(self.tacotron2.decoder, "attention"):
                self.tacotron2.decoder.attention.requires_grad_(True)
            elif hasattr(self.tacotron2, "attention"):
                self.tacotron2.attention.requires_grad_(True)
            self.tacotron2.encoder.requires_grad_(False)
            self.tacotron2.embedding.requires_grad_(False)
            print("Fine-tuning enabled for Tacotron2 decoder")

    def disable_fine_tuning(self):
        """Disable fine-tuning of Tacotron2 decoder components."""
        if self.use_tacotron2:
            self.fine_tune_decoder = False
            self.tacotron2.requires_grad_(False)
            print("Fine-tuning disabled for Tacotron2 decoder")

    def get_trainable_parameters(self):
        """Get trainable parameters for the VAE."""
        params = []

        # VAE encoder parameters
        params.extend(self.encoder_prenet.parameters())
        params.extend(self.encoder_lstm.parameters())
        params.extend(self.fc_mu.parameters())
        params.extend(self.fc_logvar.parameters())
        params.extend(self.speaker_projection.parameters())
        params.extend(self.speaker_to_decoder.parameters())

        # Tacotron2 decoder parameters (if fine-tuning is enabled)
        if self.use_tacotron2 and self.fine_tune_decoder:
            params.extend(self.tacotron2.decoder.parameters())
            params.extend(self.tacotron2.postnet.parameters())
            # Try to access attention through decoder, fallback to direct access
            if hasattr(self.tacotron2.decoder, "attention"):
                params.extend(self.tacotron2.decoder.attention.parameters())
            elif hasattr(self.tacotron2, "attention"):
                params.extend(self.tacotron2.attention.parameters())

        # Fallback decoder parameters (only if Tacotron2 is not used)
        if not self.use_tacotron2 and hasattr(self, "decoder_input_proj"):
            params.extend(self.decoder_input_proj.parameters())
            params.extend(self.decoder_lstm.parameters())
            params.extend(self.decoder_out.parameters())

        return params

    def get_parameter_groups(self, vae_lr=1e-4, decoder_lr=1e-5, postnet_lr=1e-5):
        """
        Get parameter groups with different learning rates for fine-tuning.

        Args:
            vae_lr: Learning rate for VAE components
            decoder_lr: Learning rate for Tacotron2 decoder
            postnet_lr: Learning rate for Tacotron2 postnet

        Returns:
            List of parameter groups for optimizer
        """
        param_groups = []

        # VAE encoder parameters
        vae_params = []
        vae_params.extend(self.encoder_prenet.parameters())
        vae_params.extend(self.encoder_lstm.parameters())
        vae_params.extend(self.fc_mu.parameters())
        vae_params.extend(self.fc_logvar.parameters())
        vae_params.extend(self.speaker_projection.parameters())
        vae_params.extend(self.speaker_to_decoder.parameters())

        if vae_params:
            param_groups.append(
                {"params": vae_params, "lr": vae_lr, "name": "vae_components"}
            )

        # Tacotron2 decoder parameters (if fine-tuning is enabled)
        if self.use_tacotron2 and self.fine_tune_decoder:
            # Decoder RNN and attention
            decoder_params = []
            decoder_params.extend(self.tacotron2.decoder.parameters())
            # Try to access attention through decoder, fallback to direct access
            if hasattr(self.tacotron2.decoder, "attention"):
                decoder_params.extend(self.tacotron2.decoder.attention.parameters())
            elif hasattr(self.tacotron2, "attention"):
                decoder_params.extend(self.tacotron2.attention.parameters())

            if decoder_params:
                param_groups.append(
                    {
                        "params": decoder_params,
                        "lr": decoder_lr,
                        "name": "tacotron2_decoder",
                    }
                )

            # Postnet parameters
            postnet_params = list(self.tacotron2.postnet.parameters())
            if postnet_params:
                param_groups.append(
                    {
                        "params": postnet_params,
                        "lr": postnet_lr,
                        "name": "tacotron2_postnet",
                    }
                )

        # Fallback decoder parameters (only if Tacotron2 is not used)
        if not self.use_tacotron2 and hasattr(self, "decoder_input_proj"):
            fallback_params = []
            fallback_params.extend(self.decoder_input_proj.parameters())
            fallback_params.extend(self.decoder_lstm.parameters())
            fallback_params.extend(self.decoder_out.parameters())

            if fallback_params:
                param_groups.append(
                    {
                        "params": fallback_params,
                        "lr": vae_lr,
                        "name": "fallback_decoder",
                    }
                )

        return param_groups

    def count_parameters(self):
        """Count trainable parameters in the model."""
        total_params = 0
        trainable_params = 0

        for name, param in self.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
                print(f"{name}: {param.numel():,} parameters (trainable)")
            else:
                print(f"{name}: {param.numel():,} parameters (frozen)")

        # Check for fallback decoder components that might not exist
        if not self.use_tacotron2 and not hasattr(self, "decoder_input_proj"):
            print(
                "Note: Fallback decoder components not initialized (Tacotron2 is being used)"
            )

        print(f"\nTotal parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

        return total_params, trainable_params

    def decode_fallback(self, z, T):
        """
        Fallback decoder when Tacotron2 is not available.
        z: [B, latent_dim]
        T: int, target length
        """
        if not hasattr(self, "decoder_input_proj"):
            raise RuntimeError(
                "Fallback decoder not available. Tacotron2 is being used but no text was provided."
            )

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

        if self.use_tacotron2:
            if text is not None:
                # Use Tacotron2 for decoding
                recon = self.decode_with_tacotron2(z, text)
            else:
                # Tacotron2 is enabled but no text provided
                raise ValueError(
                    "Text is required when using Tacotron2. Please provide text input."
                )
        else:
            # Use fallback decoder
            T = mel.size(2)
            recon = self.decode_fallback(z, T)

        return recon, mu, logvar

    def synthesize_with_decoder(self, text, z, max_decoder_steps=1000):
        """
        Synthesize speech using Tacotron2's standard inference with speaker conditioning.
        text: list of strings
        z: [B, latent_dim] - latent speaker representation
        max_decoder_steps: maximum number of decoder steps (not used in this simplified version)
        """
        if not self.use_tacotron2:
            raise RuntimeError("Tacotron2 not available for synthesis")

        # Prepare text input for Tacotron2
        sequences, lengths = self.tts_utils.prepare_input_sequence(text)
        sequences = sequences.to(self.device)
        lengths = lengths.to(self.device)

        # Use no_grad only if not fine-tuning
        if not self.fine_tune_decoder:
            context = torch.no_grad()
        else:
            context = torch.enable_grad()

        with context:
            # Use Tacotron2's standard inference
            try:
                # Try to get 4 values from inference
                mel_outputs, mel_outputs_postnet, _, alignments = self.tacotron2.infer(
                    sequences, lengths
                )
            except ValueError:
                # If that fails, try to get 3 values
                try:
                    mel_outputs, mel_outputs_postnet, alignments = self.tacotron2.infer(
                        sequences, lengths
                    )
                except ValueError:
                    # If that also fails, try to get 2 values
                    mel_outputs, mel_outputs_postnet = self.tacotron2.infer(
                        sequences, lengths
                    )
                    # Create dummy alignments
                    alignments = torch.zeros(
                        sequences.size(0), 1, sequences.size(1)
                    ).to(self.device)

            # Apply speaker conditioning to the mel outputs
            # Project latent speaker representation to mel dimension
            speaker_conditioning = self.speaker_projection(z)  # [B, hidden_dim]

            # Project to mel dimension for conditioning
            if speaker_conditioning.size(-1) != self.n_mels:
                spk_to_mel = nn.Linear(speaker_conditioning.size(-1), self.n_mels).to(
                    self.device
                )
                speaker_conditioning = spk_to_mel(speaker_conditioning)

            # Add speaker conditioning to mel outputs
            # Expand speaker conditioning to match mel output dimensions
            if mel_outputs_postnet.dim() == 3:  # [B, n_mels, T]
                speaker_conditioning = speaker_conditioning.unsqueeze(-1).expand(
                    -1, -1, mel_outputs_postnet.size(-1)
                )
            else:  # [B, T, n_mels]
                speaker_conditioning = speaker_conditioning.unsqueeze(1).expand(
                    -1, mel_outputs_postnet.size(1), -1
                )

            mel_outputs_conditioned = mel_outputs_postnet + speaker_conditioning

        return mel_outputs_conditioned, alignments

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
            # Use the direct speaker embedding synthesis method
            return self.synthesize_with_speaker_embedding(text, spk_emb)

        # Use the new decoder-based synthesis
        return self.synthesize_with_decoder(text, z)

    def synthesize_with_speaker_embedding(self, text, spk_emb, max_decoder_steps=1000):
        """
        Synthesize speech directly from speaker embedding without going through latent space.
        This is simpler and more direct for inference.
        text: list of strings
        spk_emb: [B, spk_emb_dim]
        """
        if not self.use_tacotron2:
            raise RuntimeError("Tacotron2 not available for synthesis")

        # Prepare text input for Tacotron2
        sequences, lengths = self.tts_utils.prepare_input_sequence(text)
        sequences = sequences.to(self.device)
        lengths = lengths.to(self.device)

        # Use no_grad only if not fine-tuning
        if not self.fine_tune_decoder:
            context = torch.no_grad()
        else:
            context = torch.enable_grad()

        with context:
            # Use Tacotron2's standard inference
            try:
                # Try to get 4 values from inference
                mel_outputs, mel_outputs_postnet, _, alignments = self.tacotron2.infer(
                    sequences, lengths
                )
            except ValueError:
                # If that fails, try to get 3 values
                try:
                    mel_outputs, mel_outputs_postnet, alignments = self.tacotron2.infer(
                        sequences, lengths
                    )
                except ValueError:
                    # If that also fails, try to get 2 values
                    mel_outputs, mel_outputs_postnet = self.tacotron2.infer(
                        sequences, lengths
                    )
                    # Create dummy alignments
                    alignments = torch.zeros(
                        sequences.size(0), 1, sequences.size(1)
                    ).to(self.device)

            # Apply speaker conditioning to the mel outputs
            # Project speaker embedding directly to mel dimension
            speaker_conditioning = self.speaker_to_decoder(spk_emb)  # [B, hidden_dim]

            # Project to mel dimension for conditioning
            if speaker_conditioning.size(-1) != self.n_mels:
                spk_to_mel = nn.Linear(speaker_conditioning.size(-1), self.n_mels).to(
                    self.device
                )
                speaker_conditioning = spk_to_mel(speaker_conditioning)

            # Add speaker conditioning to mel outputs
            # Expand speaker conditioning to match mel output dimensions
            if mel_outputs_postnet.dim() == 3:  # [B, n_mels, T]
                speaker_conditioning = speaker_conditioning.unsqueeze(-1).expand(
                    -1, -1, mel_outputs_postnet.size(-1)
                )
            else:  # [B, T, n_mels]
                speaker_conditioning = speaker_conditioning.unsqueeze(1).expand(
                    -1, mel_outputs_postnet.size(1), -1
                )

            mel_outputs_conditioned = mel_outputs_postnet + speaker_conditioning

        return mel_outputs_conditioned, alignments
