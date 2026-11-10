# SnoreDetector

SnoreDetector is a research prototype for detecting snoring from audio and bringing the detector onto an iPhone. Its goal is to support unobtrusive, privacy-friendly sleep monitoring by recording short audio segments, identifying likely snoring locally, and organizing the results into sleep sessions for later review. The repository also contains early exploratory work related to inferring sleep position from snoring sounds.

## Current progress

- A binary PyTorch CNN (`SnoreNet`) classifies one-second audio clips as snoring or non-snoring.
- Audio is converted into normalized 32 × 32 MFCC features before classification.
- The included dataset contains 1,034 labeled WAV clips (501 snoring and 533 non-snoring), together with a trained PyTorch model and normalization parameters.
- Python utilities support microphone recording, audio conversion/cropping, single-file prediction, and longer recording experiments.
- The trained network can be exported to Core ML; a converted `SnoreNet.mlmodel` is included.
- A SwiftUI iOS prototype records ten-second audio units and stores recording sessions with Core Data.

The project is still in development. The iOS recording and data model are present, but the on-device MFCC preprocessing and end-to-end Core ML prediction path are not yet complete. The separate `Recipient` iOS project is currently only an early scaffold.

## Repository layout

- `PyTorch_Model/` — dataset, feature extraction, model training, prediction, and Core ML conversion.
- `coreML_Model/` — exported Core ML model.
- `iOS app/SnoreDetector/` — main iOS recording and session-management prototype.
- `iOS app/Recipient/` — experimental companion-app scaffold.
- `long_time_predict/` — audio and outputs from continuous-recording experiments.

## Project direction

The next step is to reproduce the Python preprocessing pipeline in Swift, connect it to the bundled Core ML model, aggregate predictions over each recording session, and present useful snoring events or summaries on-device. Longer-term experiments may investigate whether snoring characteristics can also provide information about sleep position.
