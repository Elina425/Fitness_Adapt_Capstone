#!/usr/bin/env python3
"""
Demo script for pose estimation on squat dataset
Capstone Project: Exercise Quality Assessment

This script demonstrates the pose estimation pipeline
and provides quick testing capabilities.
"""

import os
import sys
import json
import time
from pathlib import Path
import numpy as np
import cv2

# Import custom utilities
from pose_estimation_utils import PoseEstimator, DatasetProcessor


def quick_demo():
    """Run a quick demonstration of pose estimation capabilities"""
    print("🎯 Pose Estimation Demo - Exercise Quality Assessment")
    print("=" * 60)
    
    # Check if video directory exists
    video_dir = Path("videos_squat")
    if not video_dir.exists():
        print("❌ Video directory 'videos_squat' not found!")
        print("   Please make sure you have the squat video dataset.")
        return
    
    # Get list of available videos
    video_files = list(video_dir.glob("*.mp4"))
    if not video_files:
        print("❌ No video files found in 'videos_squat' directory!")
        return
    
    print(f"📁 Found {len(video_files)} video files")
    
    # Initialize pose estimator
    print("\n🔧 Initializing MediaPipe pose estimator...")
    estimator = PoseEstimator('mediapipe', model_complexity=2)
    
    # Test on first video
    test_video = video_files[0]
    print(f"\n🎬 Testing on: {test_video.name}")
    
    # Extract video properties
    cap = cv2.VideoCapture(str(test_video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps
    cap.release()
    
    print(f"   Properties: {width}x{height}, {fps:.1f} FPS, {duration:.1f}s, {frame_count} frames")
    
    # Test pose estimation on sample frames
    print("\n🔄 Testing pose estimation on sample frames...")
    
    cap = cv2.VideoCapture(str(test_video))
    test_frames = 10
    successful_detections = 0
    processing_times = []
    
    for i in range(test_frames):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Skip some frames for sampling
        for _ in range(max(1, frame_count // (test_frames * 3))):
            cap.read()
        
        start_time = time.time()
        keypoints = estimator.extract_keypoints(frame)
        end_time = time.time()
        
        processing_times.append(end_time - start_time)
        
        if keypoints is not None:
            successful_detections += 1
            print(f"   Frame {i+1}: ✅ Detected {keypoints.shape[0]} keypoints in {(end_time-start_time)*1000:.1f}ms")
        else:
            print(f"   Frame {i+1}: ❌ No pose detected")
    
    cap.release()
    
    # Calculate performance metrics
    avg_time = np.mean(processing_times) if processing_times else 0
    detection_rate = successful_detections / test_frames * 100
    estimated_fps = 1.0 / avg_time if avg_time > 0 else 0
    
    print(f"\n📊 Performance Summary:")
    print(f"   Detection rate: {detection_rate:.1f}% ({successful_detections}/{test_frames})")
    print(f"   Avg processing time: {avg_time*1000:.2f}ms per frame")
    print(f"   Estimated max FPS: {estimated_fps:.1f}")
    print(f"   Real-time capable: {'✅ Yes' if estimated_fps > fps else '❌ No'}")
    
    return {
        'detection_rate': detection_rate,
        'avg_processing_time': avg_time,
        'estimated_fps': estimated_fps,
        'video_properties': {
            'width': width,
            'height': height,
            'fps': fps,
            'duration': duration,
            'frame_count': frame_count
        }
    }


def test_dataset_loading():
    """Test dataset annotation loading"""
    print("\n📚 Testing dataset annotation loading...")
    
    try:
        processor = DatasetProcessor()
        
        print("✅ Dataset annotations loaded successfully!")
        print(f"   Total videos in splits: {len(processor.train_keys) + len(processor.test_keys) + len(processor.val_keys)}")
        
        # Test getting video labels for first training video
        if processor.train_keys:
            sample_video = processor.train_keys[0]
            labels = processor.get_video_labels(sample_video)
            video_path = processor.get_video_path(sample_video)
            
            print(f"\n🔍 Sample video analysis ({sample_video}):")
            print(f"   Video exists: {'✅ Yes' if video_path and video_path.exists() else '❌ No'}")
            print(f"   Knees inward errors: {len(labels['knees_inward_errors'])} intervals")
            print(f"   Knees forward errors: {len(labels['knees_forward_errors'])} intervals") 
            print(f"   Has missing trajectory: {'⚠️ Yes' if labels['has_missing_trajectory'] else '✅ No'}")
            
            # Show error intervals if any
            if labels['knees_inward_errors']:
                print(f"   Inward error intervals: {labels['knees_inward_errors']}")
            if labels['knees_forward_errors']:
                print(f"   Forward error intervals: {labels['knees_forward_errors']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return False


def benchmark_models():
    """Benchmark different pose estimation models"""
    print("\n🏃 Running pose estimation model benchmark...")
    
    try:
        processor = DatasetProcessor()
        results = processor.benchmark_models(test_videos=2)
        
        print("\n📊 Benchmark Results:")
        print("-" * 40)
        
        for model_name, metrics in results.items():
            if 'error' in metrics:
                print(f"❌ {model_name}: {metrics['error']}")
            else:
                print(f"✅ {model_name}:")
                print(f"   Processing time: {metrics['avg_processing_time']:.2f}s")
                print(f"   Detection rate: {metrics['avg_detection_rate']:.1f}%")
                print(f"   Est. FPS: {metrics['fps_estimate']:.1f}")
        
        return results
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        return {}


def main():
    """Main demonstration function"""
    print("🚀 Starting Exercise Quality Assessment Demo")
    print("=" * 60)
    
    # Step 1: Quick pose estimation demo
    demo_results = quick_demo()
    
    # Step 2: Test dataset loading
    dataset_ok = test_dataset_loading()
    
    # Step 3: Model benchmarking (if dataset loaded successfully)
    if dataset_ok:
        benchmark_results = benchmark_models()
    else:
        print("\n⚠️ Skipping model benchmark due to dataset loading issues")
        benchmark_results = {}
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 DEMO SUMMARY")
    print("=" * 60)
    
    if demo_results:
        print(f"✅ Basic pose estimation: {demo_results['detection_rate']:.1f}% success rate")
        print(f"✅ Processing speed: {demo_results['avg_processing_time']*1000:.1f}ms per frame")
    
    if dataset_ok:
        print("✅ Dataset annotations: Successfully loaded")
        
    if benchmark_results:
        best_model = max(benchmark_results.keys(), 
                        key=lambda x: benchmark_results[x].get('avg_detection_rate', 0))
        print(f"✅ Best performing model: {best_model}")
    
    print("\n🎯 Next Steps:")
    print("1. Run full pose estimation pipeline on dataset")
    print("2. Implement biomechanical feature extraction (joint angles)")
    print("3. Train sequence classification models")
    print("4. Implement quality scoring system")
    
    print("\n📖 To run full processing:")
    print("   python pose_estimation_utils.py")
    print("\n📊 To start Jupyter analysis:")
    print("   jupyter notebook dataset_and_pose_estimation.ipynb")


if __name__ == "__main__":
    main()