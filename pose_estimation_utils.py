"""
Pose Estimation Utilities for Exercise Quality Assessment
Capstone Project Implementation

This module provides utilities for extracting pose keypoints from exercise videos
using multiple pose estimation models (MediaPipe, YOLOv11, etc.)
"""

import cv2
import numpy as np
import json
import os
from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple, Union
import mediapipe as mp
from ultralytics import YOLO
import pandas as pd
from collections import defaultdict


class PoseEstimator:
    """
    Multi-model pose estimation class supporting different pose estimation backends
    """
    
    def __init__(self, model_type='mediapipe', model_complexity=2):
        """
        Initialize pose estimator
        
        Args:
            model_type: 'mediapipe', 'yolo', or 'openpose'
            model_complexity: Model complexity (0-2 for MediaPipe)
        """
        self.model_type = model_type
        self.model_complexity = model_complexity
        self.model = None
        self.setup_model()
    
    def setup_model(self):
        """Setup the specified pose estimation model"""
        if self.model_type == 'mediapipe':
            mp_pose = mp.solutions.pose
            self.model = mp_pose.Pose(
                static_image_mode=False,
                model_complexity=self.model_complexity,
                enable_segmentation=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            print(f"✅ MediaPipe BlazePose initialized (complexity: {self.model_complexity})")
            
        elif self.model_type == 'yolo':
            try:
                self.model = YOLO('yolo11n-pose.pt')
                print("✅ YOLOv11-pose initialized")
            except Exception as e:
                print(f"❌ Failed to initialize YOLOv11: {e}")
                
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def extract_keypoints(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract keypoints from a single frame
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Array of keypoints or None if detection failed
        """
        if self.model_type == 'mediapipe':
            return self._extract_mediapipe(frame)
        elif self.model_type == 'yolo':
            return self._extract_yolo(frame)
        else:
            return None
    
    def _extract_mediapipe(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Extract keypoints using MediaPipe BlazePose"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.model.process(rgb_frame)
        
        if result.pose_landmarks:
            keypoints = []
            for landmark in result.pose_landmarks.landmark:
                keypoints.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
            return np.array(keypoints).reshape(-1, 4)  # (33, 4)
        return None
    
    def _extract_yolo(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Extract keypoints using YOLOv11"""
        if self.model is None:
            return None
            
        results = self.model(frame, verbose=False)
        if results[0].keypoints is not None and len(results[0].keypoints.data) > 0:
            keypoints = results[0].keypoints.data[0].cpu().numpy()  # First person
            return keypoints  # (17, 3) for COCO format
        return None
    
    def extract_video_keypoints(self, video_path: str, output_path: Optional[str] = None) -> Dict:
        """
        Extract keypoints from entire video
        
        Args:
            video_path: Path to input video
            output_path: Path to save keypoints (optional)
            
        Returns:
            Dictionary containing keypoints and metadata
        """
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        keypoints_sequence = []
        frame_numbers = []
        timestamps = []
        
        frame_idx = 0
        successful_detections = 0
        
        print(f"🎬 Processing video: {os.path.basename(video_path)}")
        print(f"   Frames: {frame_count}, FPS: {fps}, Resolution: {width}x{height}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Extract keypoints
            keypoints = self.extract_keypoints(frame)
            
            if keypoints is not None:
                keypoints_sequence.append(keypoints.tolist())
                successful_detections += 1
            else:
                # Add None for missing detections to maintain temporal alignment
                keypoints_sequence.append(None)
            
            frame_numbers.append(frame_idx)
            timestamps.append(frame_idx / fps)
            frame_idx += 1
            
            # Progress update
            if frame_idx % 30 == 0:
                progress = frame_idx / frame_count * 100
                print(f"   Progress: {progress:.1f}% ({successful_detections}/{frame_idx} detections)")
        
        cap.release()
        
        detection_rate = successful_detections / frame_count * 100
        print(f"✅ Completed! Detection rate: {detection_rate:.1f}% ({successful_detections}/{frame_count})")
        
        # Create result dictionary
        result = {
            'video_path': video_path,
            'video_name': os.path.basename(video_path),
            'model_type': self.model_type,
            'model_complexity': self.model_complexity,
            'keypoints_sequence': keypoints_sequence,
            'frame_numbers': frame_numbers,
            'timestamps': timestamps,
            'metadata': {
                'fps': fps,
                'frame_count': frame_count,
                'width': width,
                'height': height,
                'successful_detections': successful_detections,
                'detection_rate': detection_rate,
                'total_duration': frame_count / fps
            }
        }
        
        # Save if output path provided
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"💾 Keypoints saved to: {output_path}")
        
        return result


class DatasetProcessor:
    """
    Process the entire squat dataset for pose estimation
    """
    
    def __init__(self, video_dir='videos_squat', output_dir='keypoints_data'):
        self.video_dir = Path(video_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Load dataset annotations
        self.load_annotations()
    
    def load_annotations(self):
        """Load dataset annotations and splits"""
        print("📚 Loading dataset annotations...")
        
        try:
            with open('train_keys.json', 'r') as f:
                self.train_keys = json.load(f)
            with open('test_keys.json', 'r') as f:
                self.test_keys = json.load(f)
            with open('val_keys.json', 'r') as f:
                self.val_keys = json.load(f)
            with open('error_knees_inward.json', 'r') as f:
                self.knees_inward_errors = json.load(f)
            with open('error_knees_forward.json', 'r') as f:
                self.knees_forward_errors = json.load(f)
            with open('traj_nan.json', 'r') as f:
                self.missing_trajectories = json.load(f)
            
            print(f"   Train: {len(self.train_keys)} videos")
            print(f"   Test: {len(self.test_keys)} videos")  
            print(f"   Val: {len(self.val_keys)} videos")
            print(f"   Missing trajectories: {len(self.missing_trajectories)}")
            
        except FileNotFoundError as e:
            print(f"❌ Error loading annotations: {e}")
            raise
    
    def get_video_path(self, video_id: str) -> Optional[Path]:
        """Get full path for video ID"""
        video_path = self.video_dir / f"{video_id}.mp4"
        return video_path if video_path.exists() else None
    
    def get_video_labels(self, video_id: str) -> Dict:
        """Get quality labels for video"""
        return {
            'knees_inward_errors': self.knees_inward_errors.get(video_id, []),
            'knees_forward_errors': self.knees_forward_errors.get(video_id, []),
            'has_missing_trajectory': f"{video_id}.json" in self.missing_trajectories
        }
    
    def process_video_batch(self, video_ids: List[str], pose_estimator: PoseEstimator, 
                          batch_name: str = "batch") -> Dict:
        """Process a batch of videos"""
        print(f"\n🔄 Processing {batch_name} ({len(video_ids)} videos)")
        
        results = []
        successful_videos = 0
        
        for i, video_id in enumerate(video_ids):
            video_path = self.get_video_path(video_id)
            
            if not video_path or not video_path.exists():
                print(f"⚠️  Video not found: {video_id}")
                continue
            
            print(f"\n📹 [{i+1}/{len(video_ids)}] Processing: {video_id}")
            
            # Extract keypoints
            output_path = self.output_dir / f"{video_id}_keypoints.json"
            
            # Skip if already processed
            if output_path.exists():
                print(f"⏭️  Already processed: {video_id}")
                continue
            
            try:
                keypoint_data = pose_estimator.extract_video_keypoints(
                    str(video_path), str(output_path)
                )
                
                # Add labels
                keypoint_data['labels'] = self.get_video_labels(video_id)
                keypoint_data['split'] = self._get_video_split(video_id)
                
                results.append({
                    'video_id': video_id,
                    'status': 'success',
                    'detection_rate': keypoint_data['metadata']['detection_rate'],
                    'output_path': str(output_path)
                })
                
                successful_videos += 1
                
            except Exception as e:
                print(f"❌ Error processing {video_id}: {e}")
                results.append({
                    'video_id': video_id,
                    'status': 'error',
                    'error': str(e)
                })
        
        print(f"\n✅ {batch_name} complete: {successful_videos}/{len(video_ids)} videos processed")
        
        return {
            'batch_name': batch_name,
            'results': results,
            'successful_videos': successful_videos,
            'total_videos': len(video_ids)
        }
    
    def _get_video_split(self, video_id: str) -> str:
        """Determine which split the video belongs to"""
        if video_id in self.train_keys:
            return 'train'
        elif video_id in self.test_keys:
            return 'test'
        elif video_id in self.val_keys:
            return 'val'
        else:
            return 'unknown'
    
    def process_sample_videos(self, n_samples: int = 10, pose_estimator: PoseEstimator = None):
        """Process a sample of videos for testing"""
        if pose_estimator is None:
            pose_estimator = PoseEstimator('mediapipe')
        
        # Get sample from training set
        sample_videos = self.train_keys[:n_samples]
        return self.process_video_batch(sample_videos, pose_estimator, f"sample_{n_samples}")
    
    def benchmark_models(self, test_videos: int = 5):
        """Benchmark different pose estimation models"""
        print("🏃 Starting pose estimation model benchmark...")
        
        # Get sample videos for testing
        sample_videos = self.train_keys[:test_videos]
        
        models_to_test = [
            ('mediapipe', 1),  # MediaPipe with complexity 1
            ('mediapipe', 2),  # MediaPipe with complexity 2
            ('yolo', None)     # YOLOv11
        ]
        
        benchmark_results = {}
        
        for model_type, complexity in models_to_test:
            model_name = f"{model_type}_{complexity}" if complexity else model_type
            print(f"\n🔄 Testing {model_name}...")
            
            try:
                estimator = PoseEstimator(model_type, complexity)
                times = []
                detection_rates = []
                
                for video_id in sample_videos:
                    video_path = self.get_video_path(video_id)
                    if not video_path:
                        continue
                    
                    # Time the processing
                    start_time = time.time()
                    result = estimator.extract_video_keypoints(str(video_path))
                    end_time = time.time()
                    
                    times.append(end_time - start_time)
                    detection_rates.append(result['metadata']['detection_rate'])
                
                benchmark_results[model_name] = {
                    'avg_processing_time': np.mean(times),
                    'avg_detection_rate': np.mean(detection_rates),
                    'fps_estimate': np.mean([r['metadata']['frame_count'] for r in [result]]) / np.mean(times),
                    'model_type': model_type,
                    'complexity': complexity
                }
                
                print(f"   Avg processing time: {np.mean(times):.2f}s")
                print(f"   Avg detection rate: {np.mean(detection_rates):.1f}%")
                
            except Exception as e:
                print(f"❌ Error testing {model_name}: {e}")
                benchmark_results[model_name] = {'error': str(e)}
        
        return benchmark_results


def main():
    """Main function for testing pose estimation"""
    print("🚀 Pose Estimation for Exercise Quality Assessment")
    print("=" * 50)
    
    # Initialize dataset processor
    processor = DatasetProcessor()
    
    # Benchmark models
    print("\n1️⃣ Benchmarking pose estimation models...")
    benchmark_results = processor.benchmark_models(test_videos=3)
    
    # Display benchmark results
    print("\n📊 Benchmark Results:")
    for model_name, results in benchmark_results.items():
        if 'error' not in results:
            print(f"   {model_name}:")
            print(f"      Processing time: {results['avg_processing_time']:.2f}s")
            print(f"      Detection rate: {results['avg_detection_rate']:.1f}%")
            print(f"      Est. real-time FPS: {results['fps_estimate']:.1f}")
    
    # Process sample videos with best model
    print(f"\n2️⃣ Processing sample videos...")
    
    # Choose MediaPipe as it's typically most reliable
    estimator = PoseEstimator('mediapipe', model_complexity=2)
    sample_results = processor.process_sample_videos(n_samples=5, pose_estimator=estimator)
    
    print(f"\n✅ Sample processing complete!")
    print(f"   Successful: {sample_results['successful_videos']}/{sample_results['total_videos']}")
    
    return benchmark_results, sample_results


if __name__ == "__main__":
    benchmark_results, sample_results = main()