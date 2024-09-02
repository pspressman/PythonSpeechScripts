import os
os.environ['OPENSMILE_ROOT'] = '/Users/peterpressman/opensmile'
import whisperx
import gc 
import torch
import torchaudio
from tqdm import tqdm
# Set these to your actual paths
os.environ['OPENSMILE_ROOT'] = '/Users/peterpressman/opensmile'
os.environ['LD_LIBRARY_PATH'] = '/path/to/your/opensmile/installation/lib:' + os.environ.get('LD_LIBRARY_PATH', '')
import opensmile
import pandas as pd
import numpy as np
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
from collections import Counter
import logging
import time
import traceback
import re

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_speaker_number(speaker_label):
    # Extract numeric part from the speaker label
    match = re.search(r'\d+', speaker_label)
    return int(match.group()) if match else 0  # Return 0 if no number found

def process_audio(audio_file, device, batch_size, compute_type, hf_token, min_speakers=None, max_speakers=None):
    try:
        start_time = time.time()
        logging.info(f"Starting processing of {audio_file}")

        # Try to load the model with the specified device
        try:
            model = whisperx.load_model("large-v2", device, compute_type=compute_type)
        except ValueError as e:
            if "unsupported device mps" in str(e):
                logging.warning(f"MPS device not supported for this operation. Falling back to CPU for {audio_file}")
                device = "cpu"
                model = whisperx.load_model("large-v2", device, compute_type=compute_type)
            else:
                raise

        # Transcribe with whisperx
        logging.debug("Transcribing audio")
        audio = whisperx.load_audio(audio_file)
        result = model.transcribe(audio, batch_size=batch_size)
        logging.debug(f"Transcription completed. Number of segments: {len(result['segments'])}")
        
        del model
        gc.collect()
        torch.mps.empty_cache() if device == "mps" else torch.cuda.empty_cache()

        # Align whisper output
        logging.debug("Aligning whisper output")
        model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
        result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
        logging.debug(f"Alignment completed. Number of aligned segments: {len(result['segments'])}")
        
        del model_a
        gc.collect()
        torch.mps.empty_cache() if device == "mps" else torch.cuda.empty_cache()

        # Initialize diarization pipeline
        logging.debug("Initializing diarization pipeline")
        diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=hf_token)
        diarization_pipeline.to(torch.device(device))

        # Load audio file
        waveform, sample_rate = torchaudio.load(audio_file)

        # Run diarization with progress hook
        logging.debug("Running diarization")
        with ProgressHook() as hook:
            diarization = diarization_pipeline({"waveform": waveform, "sample_rate": sample_rate}, 
                                               num_speakers=min_speakers,
                                               min_speakers=min_speakers,
                                               max_speakers=max_speakers,
                                               hook=hook)
        logging.debug(f"Diarization completed. Number of speakers: {len(set(diarization.labels()))}")

        # Process diarization results
        diarize_segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            speaker_num = extract_speaker_number(speaker)
            segment = {
                "start": turn.start,
                "end": turn.end,
                "speaker": speaker_num
            }
            diarize_segments.append(segment)

        # Assign speakers to the whisperx segments
        diarize_df = pd.DataFrame(diarize_segments)
        try:
            result = whisperx.assign_word_speakers(diarize_df, result)
        except Exception as e:
            logging.error(f"Error in assign_word_speakers: {str(e)}")
            raise

        # Process segments to handle overlaps, gaps, and pauses
        processed_segments = []
        for i, segment in enumerate(result['segments']):
            current_segment = {
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"],
                "speaker": segment.get("speaker"),
                "type": "speech"
            }
            
            if i < len(result['segments']) - 1:
                next_segment = result['segments'][i+1]
                if current_segment["end"] > next_segment["start"]:
                    # Handle overlap
                    overlap = {
                        "start": next_segment["start"],
                        "end": min(current_segment["end"], next_segment["end"]),
                        "type": "overlap",
                        "speakers": [current_segment["speaker"], next_segment.get("speaker")],
                        "controller": next_segment.get("speaker")
                    }
                    processed_segments.append(overlap)
                    current_segment["end"] = next_segment["start"]
                elif current_segment["end"] < next_segment["start"]:
                    # Handle gap
                    gap = {
                        "start": current_segment["end"],
                        "end": next_segment["start"],
                        "type": "gap",
                        "controller": next_segment.get("speaker")
                    }
                    processed_segments.append(gap)
                elif current_segment["speaker"] == next_segment.get("speaker"):
                    # Handle pause
                    pause = {
                        "start": current_segment["end"],
                        "end": next_segment["start"],
                        "type": "pause",
                        "speaker": current_segment["speaker"]
                    }
                    processed_segments.append(pause)
            
            processed_segments.append(current_segment)

        total_time = time.time() - start_time
        logging.info(f"Total processing time for {audio_file}: {total_time:.2f} seconds")

        return processed_segments, diarization
    except Exception as e:
        logging.error(f"Error processing {audio_file}: {str(e)}")
        logging.error(traceback.format_exc())
        raise

def extract_features(audio_file, start, end, feature_set='eGeMAPSv02'):
    try:
        logging.debug(f"Extracting features from {audio_file} (start: {start}, end: {end})")
        
        # Create Smile object
        smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals,
        )
        
        # Load audio file
        waveform, sample_rate = torchaudio.load(audio_file)
        
        # Extract segment
        start_sample = int(start * sample_rate)
        end_sample = int(end * sample_rate)
        segment = waveform[:, start_sample:end_sample]
        
        # Process signal
        features = smile.process_signal(segment.numpy().flatten(), sample_rate)
        
        logging.debug(f"Features extracted. Shape: {features.shape}")
        
        return features
    except Exception as e:
        logging.error(f"Error in extract_features: {str(e)}")
        logging.error(traceback.format_exc())
        raise

def aggregate_features(feature_list):
    try:
        df = pd.concat(feature_list)
        logging.debug(f"Aggregating features. Number of feature sets: {len(feature_list)}")
        
        aggregated = df.agg(['mean', 'median', 'std', 'min', 'max'])
        aggregated = aggregated.T.add_suffix('_value').T
        
        # Calculate coefficient of variation (CoV) for non-zero means
        cov = df.std() / df.mean().replace(0, np.nan)
        aggregated.loc['cov'] = cov.fillna(0)
        
        # Calculate IQR
        q75 = df.quantile(0.75)
        q25 = df.quantile(0.25)
        aggregated.loc['iqr'] = q75 - q25
        
        logging.debug(f"Feature aggregation completed. Shape: {aggregated.shape}")
        return aggregated.T.reset_index().melt(id_vars='index')
    except Exception as e:
        logging.error(f"Error in aggregate_features: {str(e)}")
        logging.error(traceback.format_exc())
        raise
        
def main():
    start_time = time.time()
    logging.info("Starting the audio processing script")
    
    # Determine the device
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    logging.info(f"Initial device selection: {device}")
    
    batch_size = 16
    compute_type = "float32"  # Use float32 for both MPS and CPU
    
    hf_token = "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"  # Replace with your actual token
  
    audio_dir = "/Users/peterpressman/Desktop/A-Z/D/Data/CSAND/CSA/Spliced Audio/Grandfather Passage/"
    output_dir = "/Users/peterpressman/Desktop/A-Z/D/Data/CSAND/CSA/Spliced Audio/GFPOutput/"
  
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Output directory created/verified: {output_dir}")
  
    audio_files = [f for f in os.listdir(audio_dir) if f.endswith(('.mp3', '.wav', '.m4a', '.flac'))]
    logging.info(f"Found {len(audio_files)} audio files to process")
  
    for audio_file in tqdm(audio_files, desc="Processing audio files"):
        file_start_time = time.time()
        full_path = os.path.join(audio_dir, audio_file)
        logging.info(f"Processing: {audio_file}")
        
        try:
            processed_segments, diarization = process_audio(full_path, device, batch_size, compute_type, hf_token, min_speakers=2, max_speakers=5)
            
            if not processed_segments:
                logging.warning(f"No valid segments for {audio_file}. Skipping feature extraction.")
                continue

            # Extract and aggregate features
            speaker_features = {}
            non_speech_segments = []
            
            for segment in processed_segments:
                if segment['type'] == 'speech':
                    speaker = segment['speaker']
                    features = extract_features(full_path, segment['start'], segment['end'])
                    
                    if speaker not in speaker_features:
                        speaker_features[speaker] = []
                    speaker_features[speaker].append(features)
                else:
                    non_speech_segments.append(segment)

            # Aggregate features for each speaker
            for speaker, feature_list in speaker_features.items():
                aggregated_features = aggregate_features(feature_list)
                
                # Save aggregated features
                output_file = os.path.join(output_dir, f"{os.path.splitext(audio_file)[0]}_speaker{speaker}_features.csv")
                aggregated_features.to_csv(output_file, index=False)
                logging.info(f"Saved aggregated features for speaker {speaker} to {output_file}")
            
            # Save non-speech segments information
            non_speech_file = os.path.join(output_dir, f"{os.path.splitext(audio_file)[0]}_non_speech_segments.csv")
            pd.DataFrame(non_speech_segments).to_csv(non_speech_file, index=False)
            logging.info(f"Saved non-speech segments to {non_speech_file}")

            # Save diarization output in RTTM format
            rttm_file = os.path.join(output_dir, f"{os.path.splitext(audio_file)[0]}.rttm")
            with open(rttm_file, "w") as rttm:
                diarization.write_rttm(rttm)
            logging.info(f"Saved RTTM file to {rttm_file}")
            
            # Save transcription with speaker labels
            transcript_file = os.path.join(output_dir, f"{os.path.splitext(audio_file)[0]}_transcript.txt")
            with open(transcript_file, "w") as f:
                for segment in processed_segments:
                    if segment['type'] == 'speech':
                        f.write(f"Speaker {segment['speaker']}: {segment['text']}\n")
                    else:
                        f.write(f"{segment['type'].capitalize()}: {segment['start']} - {segment['end']}\n")
            logging.info(f"Saved transcript file to {transcript_file}")
            
            file_processing_time = time.time() - file_start_time
            logging.info(f"Successfully processed and saved outputs for: {audio_file}. Time taken: {file_processing_time:.2f} seconds")
        
        except Exception as e:
            logging.error(f"Error processing {audio_file}: {str(e)}")
            logging.error(traceback.format_exc())
            if "unsupported device mps" in str(e) and device != "cpu":
                logging.info(f"Retrying {audio_file} with CPU")
                try:
                    device = "cpu"
                    processed_segments, diarization = process_audio(full_path, device, batch_size, compute_type, hf_token, min_speakers=2, max_speakers=5)
                    # Re-run the feature extraction and saving code here
                    # (You may want to create a separate function for this to avoid code duplication)
                except Exception as e:
                    logging.error(f"Error processing {audio_file} with CPU: {str(e)}")
                    logging.error(traceback.format_exc())

    total_processing_time = time.time() - start_time
    logging.info(f"Audio processing script completed. Total time taken: {total_processing_time:.2f} seconds")

if __name__ == "__main__":
    main()
