# PythonSpeechScripts
# AudioPipeline Documentation

## Overview

This script implements an advanced audio processing pipeline that performs transcription, speaker diarization, and acoustic feature extraction on audio files. It uses state-of-the-art models and libraries to process multiple audio formats, generate transcripts with speaker labels, create diarization outputs, and extract acoustic features for each speaker.

## Key Components

1. **Transcription**: Uses WhisperX for accurate speech-to-text conversion.
2. **Diarization**: Employs pyannote.audio's speaker diarization model (version 3.1) from HuggingFace.
3. **Voice Activity Detection**: Utilizes pyannote.audio's segmentation model (version 3.0) from HuggingFace.
4. **Feature Extraction**: Leverages OpenSMILE for extracting acoustic features.

## Environment and Setup

- **Operating System**: macOS Sonoma (version 14.4)
- **Python Environment**: Base Conda environment
- **Execution**: Command-line or IDE compatible with base environment (Spyder not compatible)

## Dependencies

- torch (PyTorch)
- torchaudio
- whisperx
- pyannote.audio
- opensmile
- pandas
- numpy
- FFmpeg (disabled in build)

## Configuration Steps

1. **CMake Configuration**:
   - Use CCMake to modify build configuration
   - Set WITH_FFMPEG to OFF
   - Navigate to OpenSMILE build directory
   - Run `ccmake ..`
   - Toggle WITH_FFMPEG to OFF
   - Save and exit CCMake
   - Rebuild OpenSMILE

2. **Device Configuration**:
   - Priority: CUDA > MPS > CPU
   - Fallback to CPU if MPS is unsupported

3. **Hugging Face Authentication**:
   - Use a valid Hugging Face token for model access

4. **Audio Processing**:
   - Supports multiple formats: .mp3, .wav, .m4a, .flac
   - WhisperX for transcription
   - pyannote.audio for diarization and segmentation
   - OpenSMILE for feature extraction

5. **Output Handling**:
   - Aggregated features per speaker (CSV)
   - Non-speech segments (CSV)
   - Diarization output (RTTM format)
   - Transcript with speaker labels (TXT)

6. **Error Handling and Logging**:
   - Robust try-except blocks
   - Detailed logging throughout
   - Performance timing for overall script and individual files

## Installation Notes

- Avoid modifying OpenSMILE source or build files
- Use pre-built or pip-installed versions when possible
- OpenSMILE may have compatibility issues with specialized IDEs like Spyder

## Execution

Run the script from the command line in the base Conda environment. Ensure all dependencies are installed and properly configured before execution.

## Troubleshooting

- If encountering FFmpeg-related errors, verify FFmpeg is disabled in the build
- For MPS-related errors, the script will automatically fall back to CPU
- If using an IDE, ensure it's compatible with the base Conda environment

This documentation provides a comprehensive overview of the AudioPipeline, its components, setup process, and key considerations for successful execution and troubleshooting.

