"""
Particle Descriptor Extraction and Neighborhood Database

This script extracts particle-level descriptors from binary segmentation
masks and stores the results in both CSV and SQLite formats.

Main features
-------------
- Particle geometry extraction
- Coordinate conversion from pixels to nanometers
- Nearest-neighbor distance calculation
- Local neighborhood descriptor extraction
- SQLite particle and neighbor database export
- Particle-level CSV export

This script intentionally focuses only on parameter extraction and
database construction. Correlation analysis and plotting are excluded.
"""

import argparse
import json
import os
import re
import sqlite3
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

VALID_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


# ----------------------------------------
# File and parsing utilities
# ----------------------------------------
def list_images(folder: str) -> List[str]:
    """List valid image files from a directory."""
    files = [f for f in os.listdir(folder) if f.lower().endswith(VALID_EXTS)]
    return sorted(files)


def parse_step_image_id(filename: str) -> Tuple[str, str]:
    """
    Parse filenames such as '10_2509.png' into step_id='10' and image_id='2509'.
    If the pattern is not matched, return step_id='NA' and image_id=<basename>.
    """
    base = os.path.splitext(os.path.basename(filename))[0]
    match = re.match(r"^(\d+)[_](\d+)$", base)
    if match:
        return match.group(1), match.group(2)
    return "NA", base


def pixel_size_nm_from_shape(width_px: int, fov_nm: float) -> float:
    """Compute nanometers per pixel from image width and field of view."""
    if fov_nm <= 0:
        raise ValueError("fov_nm must be a positive number.")
    return float(fov_nm) / float(width_px)


def is_circle_within_image(
    x_nm: float,
    y_nm: float,
    radius_nm: float,
    fov_nm: float,
) -> bool:
    """Check whether a local neighborhood circle is fully inside the image."""
    return (
        x_nm - radius_nm >= 0.0
        and y_nm - radius_nm >= 0.0
        and x_nm + radius_nm <= fov_nm
        and y_nm + radius_nm <= fov_nm
    )


# ----------------------------------------
# Particle descriptor extraction
# ----------------------------------------
def compute_geometry(
    contour: np.ndarray,
    pixel_size_nm: float,
    min_area_px: float,
) -> Optional[Dict[str, float]]:
    """Extract particle geometry descriptors from one contour."""
    area_px = float(cv2.contourArea(contour))
    if area_px < min_area_px:
        return None

    perimeter_px = float(cv2.arcLength(contour, True))

    area_nm2 = area_px * (pixel_size_nm ** 2)
    perimeter_nm = perimeter_px * pixel_size_nm

    rect = cv2.minAreaRect(contour)
    (_, (rect_w, rect_h), _) = rect
    aspect_ratio = max(rect_w, rect_h) / max(min(rect_w, rect_h), 1.0)

    circularity = (
        (4.0 * np.pi * area_px) / (perimeter_px ** 2)
        if perimeter_px > 0
        else 0.0
    )

    if len(contour) >= 5:
        ellipse = cv2.fitEllipse(contour)
        (_, axes, _) = ellipse
        major_axis, minor_axis = max(axes), min(axes)
        eccentricity = (
            np.sqrt(1.0 - (minor_axis / major_axis) ** 2)
            if major_axis > 0
            else 0.0
        )
    else:
        eccentricity = 0.0

    hull = cv2.convexHull(contour)
    hull_area_px = float(cv2.contourArea(hull))
    hull_perimeter_px = float(cv2.arcLength(hull, True))

    solidity = area_px / hull_area_px if hull_area_px > 0 else 0.0
    convexity = hull_perimeter_px / perimeter_px if perimeter_px > 0 else 0.0

    return {
        "area_nm2": float(area_nm2),
        "perimeter_nm": float(perimeter_nm),
        "eccentricity": float(eccentricity),
        "aspect_ratio": float(aspect_ratio),
        "circularity": float(circularity),
        "solidity": float(solidity),
        "convexity": float(convexity),
        "area_px": float(area_px),
        "perimeter_px": float(perimeter_px),
    }


def extract_particles_from_mask(
    mask_path: str,
    fov_nm: float,
    min_area_px: float,
) -> Tuple[List[Dict], Optional[Tuple[int, int, float]]]:
    """Extract particle descriptors from one binary mask."""
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return [], None

    height_px, width_px = mask.shape[:2]
    pixel_size_nm = pixel_size_nm_from_shape(width_px, fov_nm)

    binary_mask = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(
        binary_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    particles = []
    for contour in contours:
        geometry = compute_geometry(contour, pixel_size_nm, min_area_px)
        if geometry is None:
            continue

        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            continue

        cx_px = float(moments["m10"] / moments["m00"])
        cy_px = float(moments["m01"] / moments["m00"])

        particles.append({
            "cx_px": cx_px,
            "cy_px": cy_px,
            "x_nm": float(cx_px * pixel_size_nm),
            "y_nm": float(cy_px * pixel_size_nm),
            "px_nm": float(pixel_size_nm),
            **geometry,
        })

    particles.sort(key=lambda item: item["area_px"], reverse=True)
    return particles, (height_px, width_px, pixel_size_nm)


def compute_nnd_nm(centroids_nm: np.ndarray) -> List[float]:
    """Compute nearest-neighbor distance for each particle."""
    n_particles = len(centroids_nm)
    if n_particles < 2:
        return [np.nan] * n_particles

    distance_matrix = cdist(centroids_nm, centroids_nm)
    np.fill_diagonal(distance_matrix, np.inf)
    return np.min(distance_matrix, axis=1).tolist()


def attach_neighborhood_descriptors(
    particles: List[Dict],
    neighbor_radius_nm: float,
    fov_nm: float,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Add NND and local-neighborhood descriptors to particle records.

    Returns
    -------
    particles:
        Particle records with geometry, coordinates, NND, and local descriptors.
    neighbor_edges:
        One row per particle-neighbor relation.
    """
    if len(particles) == 0:
        return particles, []

    centroids_nm = np.array(
        [[p["x_nm"], p["y_nm"]] for p in particles],
        dtype=float,
    )

    nnd_list = compute_nnd_nm(centroids_nm)
    for particle, nnd_nm in zip(particles, nnd_list):
        particle["nnd_nm"] = float(nnd_nm) if np.isfinite(nnd_nm) else np.nan

    distance_matrix = cdist(centroids_nm, centroids_nm)
    np.fill_diagonal(distance_matrix, np.inf)

    neighbor_edges = []

    for center_idx, center in enumerate(particles):
        center_area = float(center["area_nm2"])
        cx_nm = float(center["x_nm"])
        cy_nm = float(center["y_nm"])

        recordable = is_circle_within_image(
            x_nm=cx_nm,
            y_nm=cy_nm,
            radius_nm=neighbor_radius_nm,
            fov_nm=fov_nm,
        )

        center["neighbor_recordable"] = int(recordable)
        center["neighbor_radius_nm"] = float(neighbor_radius_nm)

        neighbors = []
        if recordable:
            neighbor_indices = np.where(distance_matrix[center_idx] <= neighbor_radius_nm)[0].tolist()
            for neighbor_idx in neighbor_indices:
                neighbor = particles[neighbor_idx]
                distance_nm = float(distance_matrix[center_idx, neighbor_idx])
                neighbors.append({
                    "neighbor_id": neighbor["particle_id"],
                    "distance_nm": distance_nm,
                })
                neighbor_edges.append({
                    "particle_id": center["particle_id"],
                    "neighbor_id": neighbor["particle_id"],
                    "distance_nm": distance_nm,
                })

        center["neighbor_count"] = int(len(neighbors))
        center["neighbors_json"] = json.dumps(neighbors, ensure_ascii=False)

        if recordable and len(neighbors) > 0:
            neighbor_ids = {item["neighbor_id"] for item in neighbors}
            neighbor_areas = np.array(
                [
                    float(p["area_nm2"])
                    for p in particles
                    if p["particle_id"] in neighbor_ids
                ],
                dtype=float,
            )

            neighbor_mean_area = float(neighbor_areas.mean())
            neighbor_area_std = float(neighbor_areas.std(ddof=0))
            neighbor_max_area = float(neighbor_areas.max())
            neighbor_min_area = float(neighbor_areas.min())

            local_grad_mean_area = float(center_area - neighbor_mean_area)
            delta_vs_nei_max = float(center_area - neighbor_max_area)
            delta_vs_nei_min = float(center_area - neighbor_min_area)

            denominator = center_area if center_area != 0 else np.nan

            center["local_grad_mean_area"] = local_grad_mean_area
            center["nei_area_std"] = neighbor_area_std
            center["delta_vs_nei_max"] = delta_vs_nei_max
            center["delta_vs_nei_min"] = delta_vs_nei_min
            center["ratio_std_over_center"] = float(neighbor_area_std / denominator)
            center["ratio_delta_max_over_center"] = float(delta_vs_nei_max / denominator)
            center["ratio_delta_min_over_center"] = float(delta_vs_nei_min / denominator)
        else:
            center["local_grad_mean_area"] = np.nan
            center["nei_area_std"] = np.nan
            center["delta_vs_nei_max"] = np.nan
            center["delta_vs_nei_min"] = np.nan
            center["ratio_std_over_center"] = np.nan
            center["ratio_delta_max_over_center"] = np.nan
            center["ratio_delta_min_over_center"] = np.nan

    return particles, neighbor_edges


# ----------------------------------------
# SQLite database
# ----------------------------------------
def init_db(db_path: str) -> sqlite3.Connection:
    """Initialize SQLite schema."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS particles (
        particle_id TEXT PRIMARY KEY,
        step_id TEXT,
        image_id TEXT,
        image_name TEXT,

        cx_px REAL,
        cy_px REAL,
        x_nm REAL,
        y_nm REAL,

        px_nm REAL,
        fov_nm REAL,

        area_nm2 REAL,
        perimeter_nm REAL,
        eccentricity REAL,
        aspect_ratio REAL,
        circularity REAL,
        solidity REAL,
        convexity REAL,
        area_px REAL,
        perimeter_px REAL,
        nnd_nm REAL,

        neighbor_recordable INTEGER,
        neighbor_count INTEGER,
        neighbor_radius_nm REAL,

        local_grad_mean_area REAL,
        nei_area_std REAL,
        delta_vs_nei_max REAL,
        delta_vs_nei_min REAL,

        ratio_std_over_center REAL,
        ratio_delta_max_over_center REAL,
        ratio_delta_min_over_center REAL,

        neighbors_json TEXT
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS neighbors (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        particle_id TEXT,
        neighbor_id TEXT,
        distance_nm REAL,
        FOREIGN KEY(particle_id) REFERENCES particles(particle_id)
    );
    """)

    cur.execute("CREATE INDEX IF NOT EXISTS idx_particles_image ON particles(image_name);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_neighbors_particle ON neighbors(particle_id);")

    conn.commit()
    return conn


def reset_db(conn: sqlite3.Connection) -> None:
    """Clear previous particle and neighbor records."""
    cur = conn.cursor()
    cur.execute("DELETE FROM neighbors;")
    cur.execute("DELETE FROM particles;")
    conn.commit()


def insert_particles(conn: sqlite3.Connection, particles: List[Dict]) -> None:
    """Insert particle records into SQLite."""
    cur = conn.cursor()

    for p in particles:
        cur.execute("""
            INSERT INTO particles(
                particle_id, step_id, image_id, image_name,
                cx_px, cy_px, x_nm, y_nm,
                px_nm, fov_nm,
                area_nm2, perimeter_nm, eccentricity, aspect_ratio,
                circularity, solidity, convexity,
                area_px, perimeter_px, nnd_nm,
                neighbor_recordable, neighbor_count, neighbor_radius_nm,
                local_grad_mean_area, nei_area_std, delta_vs_nei_max, delta_vs_nei_min,
                ratio_std_over_center, ratio_delta_max_over_center, ratio_delta_min_over_center,
                neighbors_json
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?);
        """, (
            p["particle_id"], p["step_id"], p["image_id"], p["image_name"],
            float(p["cx_px"]), float(p["cy_px"]), float(p["x_nm"]), float(p["y_nm"]),
            float(p["px_nm"]), float(p["fov_nm"]),
            float(p["area_nm2"]), float(p["perimeter_nm"]), float(p["eccentricity"]),
            float(p["aspect_ratio"]), float(p["circularity"]), float(p["solidity"]),
            float(p["convexity"]), float(p["area_px"]), float(p["perimeter_px"]),
            float(p["nnd_nm"]) if np.isfinite(p["nnd_nm"]) else None,
            int(p["neighbor_recordable"]), int(p["neighbor_count"]), float(p["neighbor_radius_nm"]),
            p["local_grad_mean_area"], p["nei_area_std"], p["delta_vs_nei_max"], p["delta_vs_nei_min"],
            p["ratio_std_over_center"], p["ratio_delta_max_over_center"], p["ratio_delta_min_over_center"],
            p["neighbors_json"],
        ))

    conn.commit()


def insert_neighbors(conn: sqlite3.Connection, neighbor_edges: List[Dict]) -> None:
    """Insert particle-neighbor relationships into SQLite."""
    cur = conn.cursor()

    for edge in neighbor_edges:
        cur.execute("""
            INSERT INTO neighbors(particle_id, neighbor_id, distance_nm)
            VALUES (?, ?, ?);
        """, (
            edge["particle_id"],
            edge["neighbor_id"],
            float(edge["distance_nm"]),
        ))

    conn.commit()


# ----------------------------------------
# Main extraction pipeline
# ----------------------------------------
def run_extraction(
    mask_dir: str,
    output_dir: str,
    fov_nm: float,
    neighbor_radius_nm: float,
    min_area_px: float,
    reset_database: bool = True,
) -> Dict:
    """Run particle descriptor extraction and database export."""
    if not os.path.isdir(mask_dir):
        raise FileNotFoundError(f"Mask directory not found: {mask_dir}")

    os.makedirs(output_dir, exist_ok=True)

    db_path = os.path.join(output_dir, "particle_db.sqlite")
    particle_csv_path = os.path.join(output_dir, "particles.csv")
    neighbor_csv_path = os.path.join(output_dir, "neighbors.csv")
    neighbor_summary_csv_path = os.path.join(output_dir, "particles_with_neighbors.csv")

    conn = init_db(db_path)
    if reset_database:
        reset_db(conn)

    mask_files = list_images(mask_dir)
    if len(mask_files) == 0:
        raise FileNotFoundError(f"No mask images found in: {mask_dir}")

    all_particles = []
    all_neighbor_edges = []

    print(f"[INFO] Mask directory: {mask_dir}")
    print(f"[INFO] Output directory: {output_dir}")
    print(f"[INFO] FOV: {fov_nm} nm")
    print(f"[INFO] Neighbor radius: {neighbor_radius_nm} nm")
    print(f"[INFO] Minimum area: {min_area_px} px")

    for filename in mask_files:
        mask_path = os.path.join(mask_dir, filename)
        step_id, image_id = parse_step_image_id(filename)

        particles, meta = extract_particles_from_mask(
            mask_path=mask_path,
            fov_nm=fov_nm,
            min_area_px=min_area_px,
        )

        if meta is None:
            print(f"[WARN] Failed to read mask: {filename}")
            continue

        if len(particles) == 0:
            print(f"[INFO] {filename}: 0 particles")
            continue

        for idx, particle in enumerate(particles, start=1):
            particle["step_id"] = step_id
            particle["image_id"] = image_id
            particle["image_name"] = filename
            particle["particle_index"] = idx
            particle["particle_id"] = f"{step_id}_{image_id}_{idx}"
            particle["fov_nm"] = float(fov_nm)

        particles, neighbor_edges = attach_neighborhood_descriptors(
            particles=particles,
            neighbor_radius_nm=neighbor_radius_nm,
            fov_nm=fov_nm,
        )

        insert_particles(conn, particles)
        insert_neighbors(conn, neighbor_edges)

        all_particles.extend(particles)
        all_neighbor_edges.extend(neighbor_edges)

        print(f"[OK] {filename}: particles={len(particles)}, neighbor_edges={len(neighbor_edges)}")

    conn.close()

    particles_df = pd.DataFrame(all_particles)
    if "neighbors_json" in particles_df.columns:
        particles_df["neighbors_json"] = particles_df["neighbors_json"].astype(str)

    particles_df.to_csv(particle_csv_path, index=False)

    neighbor_edges_df = pd.DataFrame(all_neighbor_edges)
    neighbor_edges_df.to_csv(neighbor_csv_path, index=False)

    neighbor_summary_df = particles_df[
        (particles_df["neighbor_recordable"] == 1)
        & (particles_df["neighbor_count"] > 0)
    ].copy()
    neighbor_summary_df.to_csv(neighbor_summary_csv_path, index=False)

    print(f"[SAVED] Particle CSV: {particle_csv_path}")
    print(f"[SAVED] Neighbor CSV: {neighbor_csv_path}")
    print(f"[SAVED] Neighbor summary CSV: {neighbor_summary_csv_path}")
    print(f"[SAVED] SQLite database: {db_path}")

    return {
        "db_path": db_path,
        "particle_csv": particle_csv_path,
        "neighbor_csv": neighbor_csv_path,
        "neighbor_summary_csv": neighbor_summary_csv_path,
        "n_particles": len(all_particles),
        "n_neighbor_edges": len(all_neighbor_edges),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract particle descriptors and neighborhood statistics from binary masks."
    )

    parser.add_argument("--mask_dir", default="path/to/mask_directory")
    parser.add_argument("--output_dir", default="path/to/output_directory")

    parser.add_argument(
        "--fov_nm",
        type=float,
        required=True,
        help="Field of view in nanometers. Required because this is dataset-specific.",
    )
    parser.add_argument(
        "--neighbor_radius_nm",
        type=float,
        required=True,
        help="Neighborhood radius in nanometers. Required because this is analysis-specific.",
    )
    parser.add_argument(
        "--min_area_px",
        type=float,
        default=10.0,
        help="Minimum contour area in pixels used to filter small objects.",
    )
    parser.add_argument(
        "--keep_existing_db",
        action="store_true",
        help="Do not clear existing database records before insertion.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    run_extraction(
        mask_dir=args.mask_dir,
        output_dir=args.output_dir,
        fov_nm=args.fov_nm,
        neighbor_radius_nm=args.neighbor_radius_nm,
        min_area_px=args.min_area_px,
        reset_database=not args.keep_existing_db,
    )


if __name__ == "__main__":
    main()
