# keep this app Monolithic. 
import gradio as gr
import cv2
import numpy as np
import os
import json
import re
import shutil
import logging
from logging.handlers import RotatingFileHandler
import threading
import time
import subprocess
import gc
import math
from collections import Counter, defaultdict, OrderedDict
from pathlib import Path
from datetime import datetime
from queue import Queue, Empty
from dataclasses import dataclass, asdict, field, fields
from concurrent.futures import ThreadPoolExecutor
import hashlib
from contextlib import contextmanager
import urllib.request
import yt_dlp as ytdlp
from scenedetect import detect, ContentDetector
from PIL import Image
import torch
from torchvision.ops import box_convert
from torchvision import transforms
from ultralytics import YOLO
from DAM4SAM.dam4sam_tracker import DAM4SAMTracker
from insightface.app import FaceAnalysis
from numba import njit
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io
import imagehash
import pyiqa
from grounding_dino.groundingdino.util.inference import (
    load_model as gdino_load_model,
    load_image as gdino_load_image,
    predict as gdino_predict,
)
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from functools import lru_cache

# --- Unified Logging & Configuration ---

from app.core.config import Config
from app.core.logging import UnifiedLogger, StructuredFormatter
from app.core.utils import safe_execute_with_retry, _to_json_safe, safe_resource_cleanup, sanitize_filename
from app.core.thumb_cache import ThumbnailManager
from app.domain.models import Frame, FrameMetrics, Scene, AnalysisParameters, MaskingResult
from app.io.video import VideoManager, run_scene_detection, run_ffmpeg_extraction
from app.io.frames import postprocess_mask, render_mask_overlay, rgb_to_pil, create_frame_map
from app.ml.downloads import download_model
from app.ml.face import get_face_analyzer
from app.ml.person import PersonDetector, get_person_detector
from app.ml.quality import compute_entropy
from app.ml.grounding import load_grounding_dino_model, predict_grounding_dino
from app.ml.sam_tracker import initialize_dam4sam_tracker
from app.masking.seed_selector import SeedSelector
from app.masking.propagate import MaskPropagator
from app.masking.subject_masker import SubjectMasker
from app.pipelines.base import Pipeline
from app.pipelines.extract import ExtractionPipeline
from app.pipelines.analyze import AnalysisPipeline
from app.ui.app_ui import AppUI

# --- Legacy Monolith (DEPRECATED) ---
# This file is kept for backward compatibility but should not be used
# in the new modular architecture. Use main.py and the /app/ package instead.
#
# The original monolithic app.py has been refactored into:
# - /app/core/ - Configuration, logging, utilities
# - /app/domain/ - Data models and business logic
# - /app/io/ - Video and frame I/O operations
# - /app/ml/ - Machine learning adapters
# - /app/masking/ - Subject masking and propagation
# - /app/pipelines/ - Analysis and extraction pipelines
# - /app/ui/ - Gradio user interface
# - /app/composition.py - Dependency injection
# - main.py - Application entry point

# For backward compatibility, we can still provide the main entry point
if __name__ == "__main__":
    import warnings
    warnings.warn(
        "Running app.py directly is deprecated. Use 'python main.py' instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    # Redirect to the new main.py
    import subprocess
    import sys
    from pathlib import Path
    
    main_py = Path(__file__).parent / "main.py"
    if main_py.exists():
        subprocess.run([sys.executable, str(main_py)] + sys.argv[1:])
            else:
        print("Error: main.py not found. Please use the new modular structure.")
        sys.exit(1)


