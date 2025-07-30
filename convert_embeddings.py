import numpy as np
import json
import os

# Load the numpy embeddings
embeddings_path = "src/speaker_embedding/speaker_emb_lookup.npy"
embeddings = np.load(embeddings_path, allow_pickle=True).item()

# Convert to TTS-compatible JSON format
# TTS expects: {clip_name: {"name": speaker_name, "embedding": [...]}}
tts_embeddings = {}

for key, value in embeddings.items():
    # Extract speaker name from the key (e.g., "#p307/p307_001_mic1.flac" -> "p307")
    speaker_name = key.split("/")[0].replace("#", "")

    # Get the embedding array
    embedding = value["embedding"]

    # Convert numpy array to list for JSON serialization
    embedding_list = embedding.tolist()

    # Store in TTS format: {clip_name: {"name": speaker_name, "embedding": [...]}}
    # TTS expects the full path including wav48_silence_trimmed but without .flac extension
    # Convert "#p282/p282_082_mic1.flac" to "#wav48_silence_trimmed/p282/p282_082_mic1"
    # Handle both Unix and Windows path separators
    clean_key = key.replace("#", "").replace(".flac", "")
    # Use os.path.join to handle path separators correctly for the current OS
    import os
    full_key = f"#wav48_silence_trimmed{os.sep}{clean_key}"
    tts_embeddings[full_key] = {
        "name": speaker_name,
        "embedding": embedding_list
    }

# Save as JSON
output_path = "src/speaker_embedding/speaker_emb_lookup.json"
with open(output_path, "w") as f:
    json.dump(tts_embeddings, f, indent=2)

print(f"Converted {len(embeddings)} embeddings to {len(tts_embeddings)} speakers")
print(f"Saved to: {output_path}")
print(f"Speakers: {list(tts_embeddings.keys())}")
