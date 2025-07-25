import torch
# Load the pretrained Tacotron2 model from NVIDIA's repository
tacotron2 = torch.hub.load(
    'NVIDIA/DeepLearningExamples:torchhub',
    'nvidia_tacotron2',
    pretrained=True).eval()

torch.save(tacotron2.state_dict(), 'tacotron2_pretrained.pt')
print("saved -> tacotron2_pretrained.pt")