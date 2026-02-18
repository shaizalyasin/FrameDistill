# 🎬 The Smart Frame Filter

**Video Summarization via Feature-Based Frame Filtering**

A Python tool that intelligently summarizes videos by extracting key frames using deep learning. It leverages a pre-trained **ResNet-50** to compute feature representations of sampled frames, then uses **Cosine Similarity** to filter out redundant frames keeping only the visually distinct ones.

## How It Works

```
Raw Video → Sample every Nth frame → Extract ResNet-50 features → Compare with Cosine Similarity → Keep distinct frames
```

1. **Frame Sampling** — Extracts every Nth frame from the video using OpenCV
2. **Feature Extraction** — Passes each frame through ResNet-50 (minus the final FC layer) to get a 2048-dimensional feature vector
3. **Similarity Comparison** — Computes cosine similarity between consecutive frame features
4. **Filtering** — Keeps only frames where similarity to the last kept frame falls below a configurable threshold
5. **Output** — Saves the selected key frames as images and displays a visual summary

## Example Results

| Metric | Value |
|---|---|
| Original frames | 1,131 |
| After sampling (1/10) | 114 |
| After filtering | 67 key frames |
| Overall compression | **5.92%** of original |

## Quick Start

### 1. Install Dependencies

```bash
pip install torch torchvision opencv-python matplotlib numpy Pillow
```

### 2. Add Your Video

Place any `.mp4` video file in the project directory (or update the `VIDEO_PATH` in the notebook).

### 3. Run the Notebook

```bash
jupyter notebook smart_frame_filter.ipynb
```

Run all cells — the notebook will extract frames, compute features, filter, and save the results to `output_frames/`.

## Configuration

These parameters can be adjusted in **Section 2** of the notebook:

| Parameter | Default | Description |
|---|---|---|
| `VIDEO_PATH` | `"sample_video.mp4"` | Path to the input video |
| `EVERY_N_FRAMES` | `10` | Sample every Nth frame |
| `SIMILARITY_THRESHOLD` | `0.90` | Frames below this similarity are considered "distinct" |
| `OUTPUT_DIR` | `"output_frames"` | Directory for saved key frames |

### Threshold Guide

| Threshold | Effect |
|---|---|
| `0.80` | Aggressive — only major scene changes |
| `0.90` | Balanced — good general-purpose setting |
| `0.95` | Gentle — captures subtle transitions |

## Project Structure

```
├── smart_frame_filter.ipynb   # Main notebook
├── README.md                  
├── sample_video.mp4           # Input video
└── output_frames/             # Filtered key frames (generated)
```

## Tech Stack

- **PyTorch + torchvision** — ResNet-50 pre-trained model
- **OpenCV** — Video I/O and frame extraction
- **Matplotlib** — Visualization (similarity plots, frame grids)
- **NumPy** — Array operations
- **Pillow** — Image format conversion

