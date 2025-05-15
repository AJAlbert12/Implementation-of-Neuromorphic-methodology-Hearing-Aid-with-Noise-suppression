import sys
import os
import socket
import threading
import queue
import signal
import numpy as np
import torch
import torchaudio
import sounddevice as sd
import argparse
import time

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

# ======== USER CONFIG ========
UDP_PORT = 5005
BUFFER_SIZE = 2048
SAMPLE_RATE = 16000  # Model's required sample rate
OUTPUT_WAV_RAW = "streamed_noisy_audio_1.wav"
OUTPUT_WAV_ENHANCED = "streamed_enhanced_audio_1.wav"
CHANNELS = 1
DEBUG = True  # Set to True to see detailed logs

# ========== AUDIO QUEUES ==========
audio_q = queue.Queue()
stream_buffer = []
enhanced_buffer = []
enhancer_ready = threading.Event()  # Flag to indicate enhancer is ready

# Normalize waveform using loudness normalization
def normalize_audio(waveform):
    """
    Normalize the waveform using loudness normalization.
    """
    waveform, _ = loudness_max_norm(waveform.numpy())  # Apply max loudness normalization
    waveform, _ = loudness_rms_norm(waveform)  # Apply RMS loudness normalization
    return torch.tensor(waveform, dtype=torch.float32)

# Preprocess audio chunk (numpy array)
def preprocess_audio_chunk(chunk):
    # chunk is a numpy array of float32 normalized audio
    waveform = torch.tensor(chunk, dtype=torch.float32)
    waveform = normalize_audio(waveform)
    return waveform

# Load pretrained Spiking FullSubNet model
def load_spiking_fullsubnet():
    """
    Load the pretrained Separator model.
    """
    if DEBUG:
        print("🔄 Loading Spiking FullSubNet model...")
        
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
    if DEBUG:
        print(f"📂 Loading model from: {model_path}")
        
    state_dict = torch.load(model_path, map_location="cpu")

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys and DEBUG:
        print(f"⚠️ Missing keys: {missing_keys}")
    if unexpected_keys and DEBUG:
        print(f"⚠️ Unexpected keys: {unexpected_keys}")

    model.eval()
    if DEBUG:
        print("✅ Model loaded successfully!")
    return model

# ========== UDP RECEIVER ==========
def udp_listener():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('', UDP_PORT))
    print(f"📥 Listening on UDP port {UDP_PORT}...")

    try:
        while True:
            data, _ = sock.recvfrom(BUFFER_SIZE)
            chunk = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32767.0
            audio_q.put(chunk)
            stream_buffer.append(chunk)
            if DEBUG and len(stream_buffer) % 100 == 0:
                print(f"📊 Received {len(stream_buffer)} audio chunks")
    except Exception as e:
        print(f"❌ Receiver Error: {e}")
    finally:
        sock.close()
        print("🔌 UDP socket closed.")

# ========== REALTIME ENHANCER ==========
def realtime_enhancer():
    try:
        print("🚀 Starting enhancer thread...")
        model = load_spiking_fullsubnet()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🖥️ Using device: {device}")
        model = model.to(device)
        model.eval()

        frame_size = 16384  # ~1s at 16kHz
        buffer = []
        
        # Signal that the enhancer is ready to process audio
        enhancer_ready.set()
        print("✅ Enhancer ready and waiting for audio...")

        while True:
            try:
                if audio_q.empty():
                    if DEBUG:
                        print("⏳ Waiting for audio data...")
                    time.sleep(0.1)
                    continue
                    
                data = audio_q.get(timeout=1)
                buffer.extend(data)

                if len(buffer) >= frame_size:
                    if DEBUG:
                        print(f"🔄 Processing frame of size {frame_size}")
                    
                    frame = np.array(buffer[:frame_size])
                    buffer = buffer[frame_size:]

                    # Preprocess and normalize
                    waveform = preprocess_audio_chunk(frame).unsqueeze(0).to(device)

                    with torch.no_grad():
                        enhanced, _, _, _ = model(waveform)

                    enhanced_np = enhanced.squeeze().cpu().numpy()
                    enhanced_buffer.append(enhanced_np)

                    if DEBUG:
                        print(f"✅ Enhanced frame processed, buffer size: {len(enhanced_buffer)}")

                    # Play enhanced audio in real-time
                    sd.play(enhanced_np, SAMPLE_RATE)
                
            except queue.Empty:
                if DEBUG:
                    print("⏳ Queue empty, waiting...")
                continue
            except Exception as e:
                print(f"❌ Inference Error: {e}")
                import traceback
                traceback.print_exc()
                break
    except Exception as e:
        print(f"❌ Enhancer thread error: {e}")
        import traceback
        traceback.print_exc()

# ========== SAVE AUDIO ON EXIT ==========
def save_on_exit():
    # Save noisy audio
    if stream_buffer:
        noisy = np.concatenate(stream_buffer)
        noisy_tensor = torch.tensor(noisy).unsqueeze(0)
        torchaudio.save(OUTPUT_WAV_RAW, noisy_tensor, SAMPLE_RATE)
        print(f"📼 Noisy audio saved to {OUTPUT_WAV_RAW}")
    else:
        print("⚠️ No noisy audio to save.")

    # Save enhanced audio
    if enhanced_buffer:
        enhanced = np.concatenate(enhanced_buffer)
        enhanced_tensor = torch.tensor(enhanced).unsqueeze(0)
        torchaudio.save(OUTPUT_WAV_ENHANCED, enhanced_tensor, SAMPLE_RATE)
        print(f"🎧 Enhanced audio saved to {OUTPUT_WAV_ENHANCED}")
    else:
        print("⚠️ No enhanced audio to save.")

# ========== MAIN ==========
def main():
    listener_thread = threading.Thread(target=udp_listener, daemon=True)
    enhancer_thread = threading.Thread(target=realtime_enhancer, daemon=True)

    listener_thread.start()
    enhancer_thread.start()
    
    # Wait for enhancer to be ready
    print("⏳ Waiting for enhancer to initialize...")
    enhancer_ready.wait(timeout=60)  # Wait up to 60 seconds for enhancer to be ready
    if not enhancer_ready.is_set():
        print("⚠️ Enhancer did not initialize within timeout period")

    try:
        while True:
            time.sleep(1)  # Less CPU usage than empty pass
    except KeyboardInterrupt:
        print("\n🛑 Ctrl+C pressed. Saving audio...")
        save_on_exit()

if __name__ == "__main__":
    main()
    
    
    

# import sys
# import os
# import socket
# import threading
# import queue
# import signal
# import numpy as np
# import torch
# import torchaudio
# import sounddevice as sd
# import argparse
# import time

# sys.path.append("C:/Users/91842/Downloads/spiking-fullsubnet")
# sys.path.append("C:/Users/91842/Downloads/spiking-fullsubnet/recipes/intel_ndns/spiking_fullsubnet_freeze_phase")

# # Import relevant modules
# from audiozen.acoustics.audio_feature import load_wav, stft, mag_phase, loudness_rms_norm, norm_amplitude, istft, loudness_max_norm
# from audiozen.acoustics.filterbank import bark_filter_bank
# from efficient_spiking_neuron import MemoryState, efficient_spiking_neuron
# from audiozen.models.spiking_fullsubnet.modeling_spiking_fullsubnet import SequenceModel, SubBandSequenceModel, SubbandModel, SpikingFullSubNet
# from audiozen.models.spiking_fullsubnet.discriminator import LearnableSigmoid, Discriminator
# from audiozen.models.spiking_fullsubnet.efficient_spiking_neuron import GSULayer, StackedGSU, GSUCell
# from recipes.intel_ndns.spiking_fullsubnet_freeze_phase.model_low_freq import Separator, SequenceModel, BaseModel, SubBandSequenceWrapper, SubbandModel
# from recipes.intel_ndns.spiking_fullsubnet_freeze_phase.efficient_spiking_neuron import GSULayer, StackedGSU, GSUCell
# from recipes.intel_ndns.spiking_fullsubnet_freeze_phase.discriminator import LearnableSigmoid, Discriminator

# # ======== USER CONFIG ========
# UDP_PORT = 5005
# BUFFER_SIZE = 2048
# SAMPLE_RATE = 16000  # Model's required sample rate
# OUTPUT_WAV_RAW = "streamed_noisy_audio.wav"
# OUTPUT_WAV_ENHANCED = "streamed_enhanced_audio.wav"
# CHANNELS = 1
# DEBUG = True  # Set to True to see detailed logs

# # Performance tuning parameters - optimized for lower latency
# FRAME_SIZE = 4096  # Smaller frame (0.25s at 16kHz) for lower latency
# OVERLAP = 0.25  # 25% overlap for smoother transitions with less overhead
# MAX_LATENCY = 0.5  # Increased acceptable latency threshold
# ENHANCE_EVERY_N_FRAMES = 2  # Only process every Nth frame to reduce CPU load

# # ========== AUDIO QUEUES ==========
# audio_q = queue.Queue()
# stream_buffer = []
# enhanced_buffer = []
# enhancer_ready = threading.Event()  # Flag to indicate enhancer is ready

# # Normalize waveform using peak normalization (simpler and faster)
# def normalize_audio(waveform):
#     """
#     Simple peak normalization that's faster than loudness normalization
#     """
#     waveform_np = waveform.numpy()
#     peak = np.max(np.abs(waveform_np))
#     if peak > 0.01:  # Only normalize if signal is strong enough
#         waveform_np = waveform_np / peak * 0.9
#     return torch.tensor(waveform_np, dtype=torch.float32)

# # Preprocess audio chunk (numpy array)
# def preprocess_audio_chunk(chunk):
#     # chunk is a numpy array of float32 normalized audio
#     waveform = torch.tensor(chunk, dtype=torch.float32)
#     waveform = normalize_audio(waveform)
#     return waveform

# # Load pretrained Spiking FullSubNet model
# def load_spiking_fullsubnet():
#     """
#     Load the pretrained Separator model.
#     """
#     if DEBUG:
#         print("🔄 Loading Spiking FullSubNet model...")
        
#     model = Separator(
#         sr = 16000,
#         fdrc = 0.5,
#         n_fft = 512,
#         fb_freqs = 64,
#         hop_length = 128,
#         win_length = 512,
#         num_freqs = 256,
#         sequence_model = "GSU",
#         fb_hidden_size = 320,
#         fb_output_activate_function = False,
#         freq_cutoffs = [32, 128],
#         sb_df_orders = [5, 3, 1],
#         sb_num_center_freqs = [4, 32, 64],
#         sb_num_neighbor_freqs = [15, 15, 15],
#         fb_num_center_freqs = [4, 32, 64],
#         fb_num_neighbor_freqs = [0, 0, 0],
#         sb_hidden_size = 224,
#         sb_output_activate_function = False,
#         norm_type = "offline_laplace_norm",
#         shared_weights = True,
#         bn = True,
#     )

#     model_path = "C:/Users/91842/Downloads/spiking-fullsubnet/model_zoo/intel_ndns/spike_fsb/baseline_m/checkpoints/latest/pytorch_model.bin"
#     if DEBUG:
#         print(f"📂 Loading model from: {model_path}")
        
#     state_dict = torch.load(model_path, map_location="cpu")

#     missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
#     if missing_keys and DEBUG:
#         print(f"⚠️ Missing keys: {missing_keys}")
#     if unexpected_keys and DEBUG:
#         print(f"⚠️ Unexpected keys: {unexpected_keys}")

#     model.eval()
#     if DEBUG:
#         print("✅ Model loaded successfully!")
#     return model

# # Apply crossfade to smooth transitions between audio frames
# def crossfade(prev_frame, current_frame, overlap_samples):
#     """
#     Apply linear crossfade between two audio frames to smooth transitions.
#     """
#     if prev_frame is None or len(prev_frame) == 0:
#         return current_frame
        
#     # Create fade in/out curves
#     fade_in = np.linspace(0, 1, overlap_samples)
#     fade_out = np.linspace(1, 0, overlap_samples)
    
#     # Apply crossfade to overlapping region
#     overlap_region = prev_frame[-overlap_samples:] * fade_out + current_frame[:overlap_samples] * fade_in
    
#     # Combine with non-overlapping parts
#     result = np.concatenate([
#         prev_frame[:-overlap_samples],
#         overlap_region,
#         current_frame[overlap_samples:]
#     ])
    
#     return result

# # ========== UDP RECEIVER ==========
# def udp_listener():
#     sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
#     sock.bind(('', UDP_PORT))
#     print(f"📥 Listening on UDP port {UDP_PORT}...")

#     try:
#         while True:
#             data, _ = sock.recvfrom(BUFFER_SIZE)
#             chunk = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32767.0
#             audio_q.put(chunk)
#             stream_buffer.append(chunk)
#             if DEBUG and len(stream_buffer) % 100 == 0:
#                 print(f"📊 Received {len(stream_buffer)} audio chunks")
#     except Exception as e:
#         print(f"❌ Receiver Error: {e}")
#     finally:
#         sock.close()
#         print("🔌 UDP socket closed.")

# # ========== REALTIME ENHANCER ==========
# def realtime_enhancer():
#     try:
#         print("🚀 Starting enhancer thread...")
#         model = load_spiking_fullsubnet()
#         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         print(f"🖥️ Using device: {device}")
#         model = model.to(device)
#         model.eval()

#         overlap_samples = int(FRAME_SIZE * OVERLAP)
#         buffer = []
#         prev_enhanced = None
#         frame_counter = 0
        
#         # Signal that the enhancer is ready to process audio
#         enhancer_ready.set()
#         print("✅ Enhancer ready and waiting for audio...")

#         # Don't open audio stream here as it causes issues
#         # We'll use sd.play() for output instead

#         while True:
#             try:
#                 if audio_q.empty():
#                     if DEBUG:
#                         print("⏳ Waiting for audio data...")
#                     time.sleep(0.01)
#                     continue
                    
#                 data = audio_q.get(timeout=0.5)
#                 buffer.extend(data)

#                 # Process when we have enough data
#                 if len(buffer) >= FRAME_SIZE:
#                     frame_counter += 1
                    
#                     # Extract a frame while keeping overlap with previous frame
#                     frame = np.array(buffer[:FRAME_SIZE])
#                     buffer = buffer[FRAME_SIZE - overlap_samples:]  # Keep overlap for next frame

#                     # Skip processing some frames for better performance
#                     if frame_counter % ENHANCE_EVERY_N_FRAMES != 0:
#                         # Save the raw frame (no enhancement) but still add to buffer
#                         # This reduces latency but still maintains audio continuity
#                         enhanced_buffer.append(frame)
#                         prev_enhanced = frame
#                         continue

#                     process_start = time.time()
                    
#                     # Simple voice activity detection - only process if frame has energy
#                     frame_energy = np.mean(np.abs(frame))
#                     if frame_energy < 0.01:  # Low energy threshold
#                         # Just pass through frames with very little energy
#                         enhanced_buffer.append(frame)
#                         prev_enhanced = frame
#                         if DEBUG:
#                             print("⏩ Low energy frame, skipping enhancement")
#                         continue

#                     # Preprocess and normalize
#                     waveform = preprocess_audio_chunk(frame).unsqueeze(0).to(device)

#                     with torch.no_grad():
#                         enhanced, _, _, _ = model(waveform)

#                     # Post-processing for better voice clarity
#                     enhanced_np = enhanced.squeeze().cpu().numpy()
                    
#                     # Apply crossfade to reduce boundary artifacts if we have a previous frame
#                     if prev_enhanced is not None:
#                         enhanced_np = crossfade(prev_enhanced, enhanced_np, overlap_samples)
                    
#                     # Apply light compression to even out the audio dynamics
#                     threshold = 0.3
#                     ratio = 1.5  # Gentler ratio
#                     makeup_gain = 1.1  # Less aggressive makeup gain
                    
#                     # Simple compressor (faster implementation)
#                     above_threshold = enhanced_np > threshold
#                     enhanced_np[above_threshold] = threshold + (enhanced_np[above_threshold] - threshold) / ratio
#                     enhanced_np = enhanced_np * makeup_gain
                    
#                     # Clip to prevent distortion
#                     enhanced_np = np.clip(enhanced_np, -0.95, 0.95)
                    
#                     # Store for next crossfade
#                     prev_enhanced = enhanced_np.copy()
                    
#                     # Save to buffer for file output
#                     enhanced_buffer.append(enhanced_np)

#                     if DEBUG:
#                         process_time = time.time() - process_start
#                         latency = process_time + (FRAME_SIZE / SAMPLE_RATE)
#                         print(f"✅ Enhanced frame processed in {process_time:.3f}s, latency: {latency:.3f}s")
#                         print(f"🔊 Enhanced buffer size: {len(enhanced_buffer)}")
                        
#                         if latency > MAX_LATENCY:
#                             print(f"⚠️ High latency detected: {latency:.3f}s exceeds {MAX_LATENCY}s threshold")

#                     # Play enhanced audio - use a more reliable method instead of output_stream
#                     try:
#                         # Convert to int16 for sd.play to avoid wave header issues
#                         play_data = (enhanced_np * 32767).astype(np.int16)
#                         sd.play(play_data, SAMPLE_RATE)
#                     except Exception as e:
#                         print(f"❌ Audio playback error: {e}")
                
#             except queue.Empty:
#                 if DEBUG:
#                     print("⏳ Queue empty, waiting...")
#                 continue
#             except Exception as e:
#                 print(f"❌ Inference Error: {e}")
#                 import traceback
#                 traceback.print_exc()
#                 break
                
#     except Exception as e:
#         print(f"❌ Enhancer thread error: {e}")
#         import traceback
#         traceback.print_exc()

# # ========== SAVE AUDIO FILES ==========
# def save_on_exit():
#     try:
#         # Save noisy audio
#         if stream_buffer:
#             print("📼 Saving noisy audio...")
#             noisy = np.concatenate(stream_buffer)
#             # Convert to int16 format (PCM) for better compatibility
#             noisy_int16 = (noisy * 32767).astype(np.int16)
#             noisy_tensor = torch.tensor(noisy_int16, dtype=torch.int16).unsqueeze(0)
#             torchaudio.save(OUTPUT_WAV_RAW, noisy_tensor, SAMPLE_RATE)
#             print(f"📼 Noisy audio saved to {OUTPUT_WAV_RAW}")
#         else:
#             print("⚠️ No noisy audio to save.")

#         # Save enhanced audio
#         if enhanced_buffer:
#             print("🎧 Saving enhanced audio...")
#             enhanced = np.concatenate(enhanced_buffer)
#             # Convert to int16 format (PCM) for better compatibility
#             enhanced_int16 = (enhanced * 32767).astype(np.int16)
#             enhanced_tensor = torch.tensor(enhanced_int16, dtype=torch.int16).unsqueeze(0)
#             torchaudio.save(OUTPUT_WAV_ENHANCED, enhanced_tensor, SAMPLE_RATE)
#             print(f"🎧 Enhanced audio saved to {OUTPUT_WAV_ENHANCED}")
#         else:
#             print("⚠️ No enhanced audio to save.")
#     except Exception as e:
#         print(f"❌ Error saving audio: {e}")
#         import traceback
#         traceback.print_exc()

# # ========== MAIN ==========
# def main():
#     listener_thread = threading.Thread(target=udp_listener, daemon=True)
#     enhancer_thread = threading.Thread(target=realtime_enhancer, daemon=True)

#     listener_thread.start()
#     enhancer_thread.start()
    
#     # Wait for enhancer to be ready
#     print("⏳ Waiting for enhancer to initialize...")
#     enhancer_ready.wait(timeout=60)  # Wait up to 60 seconds for enhancer to be ready
#     if not enhancer_ready.is_set():
#         print("⚠️ Enhancer did not initialize within timeout period")

#     try:
#         while True:
#             time.sleep(0.1)  # More responsive than 1 second sleep
#     except KeyboardInterrupt:
#         print("\n🛑 Ctrl+C pressed. Saving audio...")
#         save_on_exit()

# if __name__ == "__main__":
#     main()