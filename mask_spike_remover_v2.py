import torch
import numpy as np
import cv2

class MaskSpikeRemoverV2:
    """
    V2: Erode spikes -> Hough edge detection -> Draw clean ellipse with feathered edges.

    Processing flow:
    1. Erode mask to remove spikes
    2. Edge detection (Canny)
    3. Hough transform to find straight lines
    4. Compute intersection points to get corners
    5. Draw clean ellipse with Gaussian blur feathering
    6. Optional frame smoothing across batch
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "erode_radius": ("FLOAT", {
                    "default": 3.0,
                    "min": 0.0,
                    "max": 20.0,
                    "step": 0.5,
                    "tooltip": "Erosion radius to remove spikes"
                }),
                "min_area": ("INT", {
                    "default": 100,
                    "min": 0,
                    "max": 10000,
                    "step": 1,
                    "tooltip": "Minimum mask area to be considered valid"
                }),
                "edge_blur": ("FLOAT", {
                    "default": 3.0,
                    "min": 0.0,
                    "max": 200.0,
                    "step": 0.5,
                    "tooltip": "Edge blur radius for feathering"
                }),
                "frame_smooth": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 30,
                    "step": 1,
                    "tooltip": "Frame smoothing (0 = off)"
                }),
            },
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "process_mask"
    CATEGORY = "mask"
    DESCRIPTION = "V2: Erode spikes -> Hough edge detection -> Draw clean ellipse"

    def process_mask(self, mask, erode_radius, min_area, edge_blur, frame_smooth):
        batch_size, height, width = mask.shape
        processed_masks = []

        # Step 1: Extract bbox for each frame using edge detection on eroded mask
        bboxes = []
        for b in range(batch_size):
            mask_np = mask[b].cpu().numpy()

            # Step 1a: Erode to remove spikes (strong erosion)
            binary = (mask_np > 0.5).astype(np.uint8)
            if erode_radius > 0:
                kernel_size = max(3, int(erode_radius * 2 + 1) | 1)
                kernel = np.ones((kernel_size, kernel_size), np.uint8)
                binary = cv2.erode(binary, kernel, iterations=3)

            # Step 1b: Edge detection
            edges = cv2.Canny(binary, 50, 150)

            # Step 1c: Hough transform to find straight lines
            lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=50)

            if lines is None or len(lines) == 0:
                # Fallback: use bounding box from binary
                rows = np.any(binary, axis=1)
                cols = np.any(binary, axis=0)
                if np.any(rows) and np.any(cols):
                    y_min, y_max = np.where(rows)[0][[0, -1]]
                    x_min, x_max = np.where(cols)[0][[0, -1]]
                    area = (x_max - x_min) * (y_max - y_min)
                    if area >= min_area:
                        bboxes.append((x_min, y_min, x_max, y_max))
                    else:
                        bboxes.append(None)
                else:
                    bboxes.append(None)
                continue

            # Step 1d: Separate horizontal and vertical lines
            h_lines = []
            v_lines = []

            for line in lines:
                rho, theta = line[0]
                if theta < np.pi/4 or theta > 3*np.pi/4:
                    h_lines.append((rho, theta))
                else:
                    v_lines.append((rho, theta))

            # Step 1e: Cluster lines and find dominant ones
            def cluster_lines(lines, angle_threshold=0.1):
                if len(lines) == 0:
                    return []
                clusters = []
                for rho, theta in lines:
                    found = False
                    for i, (crho, ctheta, _) in enumerate(clusters):
                        if abs(theta - ctheta) < angle_threshold:
                            clusters[i] = (crho, ctheta, clusters[i][2] + 1)
                            found = True
                            break
                    if not found:
                        clusters.append((rho, theta, 1))
                result = {}
                for rho, theta, count in clusters:
                    key = round(theta, 1)
                    if key not in result or count > result[key][1]:
                        result[key] = ((rho, theta), count)
                return [v[0] for v in result.values()]

            h_lines_clustered = cluster_lines(h_lines)
            v_lines_clustered = cluster_lines(v_lines)

            # Step 1f: Compute intersection points
            def line_intersection(line1, line2):
                rho1, theta1 = line1
                rho2, theta2 = line2
                A = np.array([[np.cos(theta1), np.sin(theta1)],
                              [np.cos(theta2), np.sin(theta2)]])
                b = np.array([rho1, rho2])
                try:
                    point = np.linalg.solve(A, b)
                    return (int(point[0]), int(point[1]))
                except:
                    return None

            corners = []
            for hl in h_lines_clustered[:2]:
                for vl in v_lines_clustered[:2]:
                    pt = line_intersection(hl, vl)
                    if pt:
                        corners.append(pt)

            if len(corners) >= 4:
                xs = [c[0] for c in corners]
                ys = [c[1] for c in corners]
                x_min, x_max = min(xs), max(xs)
                y_min, y_max = min(ys), max(ys)
                area = (x_max - x_min) * (y_max - y_min)
                if area >= min_area:
                    bboxes.append((x_min, y_min, x_max, y_max))
                else:
                    bboxes.append(None)
            else:
                bboxes.append(None)

        # Step 2: Smooth bbox across frames (preserve original empty frames)
        if frame_smooth > 0 and batch_size > 1:
            smooth_bboxes = []
            half_window = frame_smooth // 2

            for i in range(batch_size):
                if bboxes[i] is None:
                    smooth_bboxes.append(None)
                    continue

                x_min, y_min, x_max, y_max = bboxes[i]
                area = (x_max - x_min) * (y_max - y_min)

                if area < min_area:
                    smooth_bboxes.append(None)
                    continue

                valid_bboxes = []
                for j in range(max(0, i - half_window), min(batch_size, i + half_window + 1)):
                    if bboxes[j] is not None:
                        bx_min, by_min, bx_max, by_max = bboxes[j]
                        barea = (bx_max - bx_min) * (by_max - by_min)
                        if barea >= min_area:
                            valid_bboxes.append(bboxes[j])

                if len(valid_bboxes) > 0:
                    avg_bbox = tuple(
                        int(sum(v[j] for v in valid_bboxes) / len(valid_bboxes))
                        for j in range(4)
                    )
                    smooth_bboxes.append(avg_bbox)
                else:
                    smooth_bboxes.append(None)

            bboxes = smooth_bboxes

        # Step 3: Draw ellipse with feathered edges
        for b in range(batch_size):
            if bboxes[b] is None:
                processed_masks.append(torch.zeros((height, width), dtype=torch.float32))
                continue

            x_min, y_min, x_max, y_max = bboxes[b]

            cx = (x_min + x_max) // 2
            cy = (y_min + y_max) // 2
            rx = (x_max - x_min) // 2
            ry = (y_max - y_min) // 2

            new_mask = np.zeros((height, width), dtype=np.float32)
            cv2.ellipse(new_mask, (cx, cy), (rx, ry), 0, 0, 360, 1.0, -1)

            if edge_blur > 0:
                new_mask = cv2.GaussianBlur(new_mask, (0, 0), sigmaX=edge_blur, sigmaY=edge_blur)

            processed_masks.append(torch.from_numpy(new_mask).float())

        result = torch.stack(processed_masks, dim=0)
        return (result,)


# Node class mappings
NODE_CLASS_MAPPINGS = {
    "MaskSpikeRemoverV2": MaskSpikeRemoverV2,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MaskSpikeRemoverV2": "Mask Spike Remover V2",
}
