import sys
import os
import socket
import struct
import argparse
import threading
import pygame
import torch
import torchaudio
import numpy as np
# from pydub import AudioSegment
# from pydub.playback import play

# Import your enhancement modules
sys.path.append("C:/Users/91842/Downloads/spiking-fullsubnet")
sys.path.append("C:/Users/91842/Downloads/spiking-fullsubnet/recipes/intel_ndns/spiking_fullsubnet_freeze_phase")

# Import relevant modules
from audiozen.acoustics.audio_feature import load_wav, stft, mag_phase, loudness_rms_norm, norm_amplitude, istft, loudness_max_norm
from audiozen.acoustics.filterbank import bark_filter_bank
from efficient_spiking_neuron import MemoryState, efficient_spiking_neuron
from audiozen.models.spiking_fullsubnet.modeling_spiking_fullsubnet import SequenceModel, SubBandSequenceModel, SubbandModel, SpikingFullSubNet
from audiozen.models.spiking_fullsubnet.discriminator import LearnableSigmoid, Discriminator
from audiozen.models.spiking_fullsubnet.efficient_spiking_neuron import GSULayer, StackedGSU, GSUCell
from recipes.intel_ndns.spiking_fullsubnet_freeze_phase.model_low_freq import Separator, SequenceModel, BaseModel, SubBandSequenceWrapper, SubbandModel
from recipes.intel_ndns.spiking_fullsubnet_freeze_phase.efficient_spiking_neuron import GSULayer, StackedGSU, GSUCell
from recipes.intel_ndns.spiking_fullsubnet_freeze_phase.discriminator import LearnableSigmoid, Discriminator

class AudioReceiver:
    def __init__(self, port=5000, save_dir="./received_audio"):
        self.port = port
        self.save_dir = save_dir
        self.socket = None
        self.is_running = False
        
        # Create save directory if it doesn't exist
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Initialize the enhancement model
        self.model = self.load_spiking_fullsubnet()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(device)
        self.device = device
        print(f"Using device: {device}")
        
        # Initialize pygame for audio playback
        pygame.mixer.init(frequency=16000, size=-16, channels=1)
        
    def load_spiking_fullsubnet(self):
        """
        Load the pretrained Separator model.
        """
        model = Separator(
        sr = 16000,
        fdrc = 0.5,
        n_fft = 512,
        fb_freqs = 64,
        hop_length = 128,
        win_length = 512,
        num_freqs = 256,
        sequence_model = "GSU",
        fb_hidden_size = 320,
        fb_output_activate_function = False,
        freq_cutoffs = [32, 128],
        sb_df_orders = [5, 3, 1],
        sb_num_center_freqs = [4, 32, 64],
        sb_num_neighbor_freqs = [15, 15, 15],
        fb_num_center_freqs = [4, 32, 64],
        fb_num_neighbor_freqs = [0, 0, 0],
        sb_hidden_size = 224,
        sb_output_activate_function = False,
        norm_type = "offline_laplace_norm",
        shared_weights = True,
        bn = True,
        )

        model_path = "C:/Users/91842/Downloads/spiking-fullsubnet/model_zoo/intel_ndns/spike_fsb/baseline_m/checkpoints/latest/pytorch_model.bin"
        state_dict = torch.load(model_path, map_location="cpu")

        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if missing_keys:
            print(f"⚠️ Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"⚠️ Unexpected keys: {unexpected_keys}")

        model.eval()
        return model
        
    def normalize_audio(self, waveform):
        """
        Normalize the waveform using loudness normalization.
        """
        waveform, _ = loudness_max_norm(waveform.numpy())  # Apply max loudness normalization
        waveform, _ = loudness_rms_norm(waveform)  # Apply RMS loudness normalization
        return torch.tensor(waveform, dtype=torch.float32)

    def preprocess_audio(self, audio_path):
        print(f"🔊 Processing audio: {audio_path}")
        waveform, sample_rate = torchaudio.load(audio_path)

        # Resample if needed
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)

        # Normalize
        waveform = self.normalize_audio(waveform)
        return waveform
        
    def enhance_audio(self, input_path, output_path):
        """Process the audio file with the enhancement model"""
        print(f"Enhancing audio: {input_path}")
        
        # Preprocess the audio
        waveform = self.preprocess_audio(input_path)
        waveform = waveform.to(self.device)
        
        # Run the enhancement model
        with torch.no_grad():
            enhanced_y, _, _, _ = self.model(waveform)
            
        # Save the enhanced audio
        enhanced_y = enhanced_y.cpu()
        if enhanced_y.ndim == 1:
            enhanced_y = enhanced_y.unsqueeze(0)
            
        torchaudio.save(output_path, enhanced_y, 16000)
        print(f"✅ Enhanced audio saved to: {output_path}")
        
        return enhanced_y
    
    def play_audio(self, audio_path):
        """Play the audio file using pygame"""
        print(f"Playing audio: {audio_path}")
        pygame.mixer.music.load(audio_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
            
    def start(self):
        """Start the server to receive audio files"""
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind(('0.0.0.0', self.port))
        self.socket.listen(5)
        
        self.is_running = True
        print(f"Listening for incoming audio files on port {self.port}...")
        
        try:
            while self.is_running:
                client_socket, addr = self.socket.accept()
                print(f"Connection from {addr}")
                
                # Start a new thread to handle this client
                client_thread = threading.Thread(
                    target=self.handle_client,
                    args=(client_socket, addr)
                )
                client_thread.daemon = True
                client_thread.start()
                
        except KeyboardInterrupt:
            print("Shutting down server...")
        finally:
            if self.socket:
                self.socket.close()
                
    def handle_client(self, client_socket, addr):
        """Handle an incoming client connection"""
        try:
            # Receive file size (8 bytes, unsigned long long)
            size_data = client_socket.recv(8)
            file_size = struct.unpack("!Q", size_data)[0]
            print(f"Receiving file of size: {file_size} bytes")
            
            # Generate filenames
            timestamp = int(pygame.time.get_ticks())
            received_filename = os.path.join(self.save_dir, f"received_{timestamp}.wav")
            enhanced_filename = os.path.join(self.save_dir, f"enhanced_{timestamp}.wav")
            
            # Receive and save the file
            bytes_received = 0
            with open(received_filename, 'wb') as f:
                while bytes_received < file_size:
                    chunk = client_socket.recv(min(4096, file_size - bytes_received))
                    if not chunk:
                        break
                    f.write(chunk)
                    bytes_received += len(chunk)
                    print(f"Received {bytes_received}/{file_size} bytes ({bytes_received/file_size*100:.1f}%)")
                    
            print(f"File received and saved to {received_filename}")
            
            # Process the audio with the enhancement model
            enhanced_audio = self.enhance_audio(received_filename, enhanced_filename)
            
            # Play the enhanced audio
            self.play_audio(enhanced_filename)
            
        except Exception as e:
            print(f"Error handling client: {e}")
        finally:
            client_socket.close()
            
def main():
    parser = argparse.ArgumentParser(description="Receive and enhance audio files")
    parser.add_argument("--port", type=int, default=5000, help="Port to listen on")
    parser.add_argument("--save-dir", default="./received_audio", help="Directory to save audio files")
    
    args = parser.parse_args()
    
    receiver = AudioReceiver(port=args.port, save_dir=args.save_dir)
    receiver.start()

if __name__ == "__main__":
    main()