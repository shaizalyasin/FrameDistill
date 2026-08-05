# FrameDistill: Feature-Based Video Summarization

A Python tool for extracting representative frames from videos using deep visual features and similarity-based filtering.

The project uses a pre-trained **ResNet-50** model to extract feature representations from sampled video frames. These features are then compared using **cosine similarity** to identify visually similar frames and keep only the frames that contain meaningful visual changes.

The goal is to reduce the number of frames while preserving the important visual information of the original video.

---

## Overview

Long videos often contain many consecutive frames with little visual change. Instead of processing or storing every frame, this project creates a compact summary by selecting only representative frames.

The pipeline is:

```
Input Video
    ↓
Sample Frames
    ↓
Extract ResNet-50 Features
    ↓
Calculate Feature Similarity
    ↓
Remove Redundant Frames
    ↓
Save Key Frames
```

---

## How It Works

### 1. Frame Sampling

The video is first sampled using OpenCV by extracting every Nth frame.

This reduces the number of frames that need to be processed while maintaining enough information about scene changes.

Example:

```
Original video: 1131 frames
Sampling interval: every 10th frame
Remaining frames: 114
```

---

### 2. Feature Extraction

Each sampled frame is passed through a pre-trained **ResNet-50** model.

The final classification layer is removed, and the output feature vector from the network is used as a visual representation of the frame.

Each frame is represented as:

```
2048-dimensional feature vector
```

These features capture high-level visual information such as objects, shapes, and scene characteristics.

---

### 3. Similarity Calculation

The similarity between frames is measured using cosine similarity.

Frames with highly similar feature representations are considered redundant.

```
Cosine similarity → higher value = more visually similar
```

---

### 4. Frame Filtering

A frame is kept only when its similarity with the previously selected frame falls below the chosen threshold.

This allows the algorithm to preserve scene changes while removing unnecessary duplicates.

---

### 5. Output

The selected frames are:

- saved as image files,
- displayed as a visual summary,
- stored in the output directory for further use.

---

## Example Results

| Metric | Value |
|---|---:|
| Original frames | 1,131 |
| After sampling (1/10) | 114 |
| After similarity filtering | 67 key frames |
| Final size compared to original | 5.92% |

The filtering step reduces the number of frames while keeping visually distinct content.

---

## Quick Start

### 1. Install Dependencies

```bash
pip install torch torchvision opencv-python matplotlib numpy Pillow
```

---

### 2. Add a Video

Place an `.mp4` video file in the project directory.

Alternatively, update the `VIDEO_PATH` variable in the notebook:

```python
VIDEO_PATH = "sample_video.mp4"
```

---

### 3. Run the Notebook

Start Jupyter Notebook:

```bash
jupyter notebook frame_distill.ipynb
```

Run all cells.

The notebook will:

1. load the video,
2. extract sampled frames,
3. compute ResNet-50 features,
4. compare frame similarities,
5. save the selected key frames.

The output will be stored in:

```
output_frames/
```

---

## Configuration

The main parameters can be adjusted directly in the notebook.

| Parameter | Default | Description |
|---|---|---|
| `VIDEO_PATH` | `"sample_video.mp4"` | Input video path |
| `EVERY_N_FRAMES` | `10` | Frame sampling interval |
| `SIMILARITY_THRESHOLD` | `0.90` | Similarity threshold for filtering |
| `OUTPUT_DIR` | `"output_frames"` | Directory for selected frames |

---

## Similarity Threshold

The threshold controls how aggressively frames are filtered.

| Threshold | Behaviour |
|---|---|
| `0.80` | Strong filtering, keeps only major scene changes |
| `0.90` | Balanced setting for general use |
| `0.95` | Less filtering, keeps more subtle transitions |

A lower threshold produces fewer frames, while a higher threshold preserves more visual details.

---

## Project Structure

```
.
├── frame_distill.ipynb        # Main implementation notebook
├── README.md
├── sample_video.mp4           # Example input video
└── output_frames/             # Generated key frames
```

---

## Technologies Used

- **PyTorch + torchvision**  
  Pre-trained ResNet-50 model for extracting visual features.

- **OpenCV**  
  Video loading and frame extraction.

- **NumPy**  
  Numerical operations and feature handling.

- **Matplotlib**  
  Visualization of similarity scores and selected frames.

- **Pillow**  
  Image processing and format conversion.

---

## Possible Extensions

Some possible improvements for future versions:

- Use temporal information in addition to visual similarity.
- Replace fixed sampling with adaptive frame selection.
- Support multiple video formats.
- Add automatic scene boundary detection.
- Compare different feature extraction models.
