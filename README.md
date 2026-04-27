# ComfyUI Mask Spike Remover V2

A ComfyUI custom node that removes spikes from masks and converts them to smooth elliptical masks with feathered edges.

## Features

- **Spike Removal**: Uses erosion to remove spikes from masks
- **Edge Detection**: Uses Canny edge detection and Hough transform to find the main edges
- **Clean Ellipse**: Draws a clean ellipse based on detected corners
- **Feathered Edges**: Applies Gaussian blur for smooth edge feathering
- **Frame Smoothing**: Optional smoothing across frames to prevent jumping
- **Area Filter**: Filters out small masks with configurable minimum area

## Installation

Copy `mask_spike_remover_v2.py` to your ComfyUI `custom_nodes` directory:

```bash
# Windows
copy mask_spike_remover_v2.py C:\aki1.7\ComfyUI-aki-v1.7\ComfyUI\custom_nodes\

# Linux/Mac
cp mask_spike_remover_v2.py /path/to/ComfyUI/custom_nodes/
```

Then restart ComfyUI.

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| mask | MASK | Required | Input mask |
| erode_radius | FLOAT | 3.0 | Erosion radius to remove spikes (0-20) |
| min_area | INT | 100 | Minimum mask area to be considered valid (0-10000) |
| edge_blur | FLOAT | 3.0 | Edge blur radius for feathering (0-50) |
| frame_smooth | INT | 0 | Frame smoothing (0 = off, higher = smoother) |

## Processing Flow

1. **Erode**: Apply erosion to remove spikes from the input mask
2. **Edge Detection**: Use Canny edge detection
3. **Hough Transform**: Find straight lines in the edge image
4. **Corner Detection**: Compute line intersections to find corners
5. **Ellipse Drawing**: Draw a clean ellipse within the detected boundaries
6. **Feathering**: Apply Gaussian blur for smooth edges
7. **Frame Smoothing**: Optionally smooth across frames

## Usage

Search for "Mask Spike Remover V2" in ComfyUI's node search.

## License

MIT
