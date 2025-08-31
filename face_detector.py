#!/usr/bin/env python3
"""
MediaPipe Face Landmark Detector (w8a8 quantized) for Raspberry Pi Zero 2 W
Optimized implementation with 468 facial landmarks using quantized model
Fixed version that handles set_num_threads compatibility issues
"""

import cv2
import numpy as np
import time
import sys
import os
from pathlib import Path
from threading import Thread
from queue import Queue
import argparse

# Check if we're in virtual environment
def check_venv():
    """Check if running in virtual environment"""
    if sys.prefix == sys.base_prefix:
        print("WARNING: Not running in virtual environment!")
        print("Please activate with: source face_env/bin/activate")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)

check_venv()

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    print("Error: tflite-runtime not installed")
    print("With virtual env activated, run: pip install tflite-runtime")
    sys.exit(1)

# MediaPipe face landmark connections for drawing
FACE_CONNECTIONS = [
    # Lips outline
    (61, 84), (84, 17), (17, 314), (314, 405), (405, 320), (320, 307),
    (307, 375), (375, 308), (308, 324), (324, 318), (318, 402), (402, 317),
    (317, 14), (14, 87), (87, 178), (178, 88), (88, 95), (95, 61),
    # Left eye
    (33, 7), (7, 163), (163, 144), (144, 145), (145, 153), (153, 154),
    (154, 155), (155, 133), (133, 173), (173, 157), (157, 158), (158, 159),
    (159, 160), (160, 161), (161, 246), (246, 33),
    # Right eye
    (263, 249), (249, 390), (390, 373), (373, 374), (374, 380), (380, 381),
    (381, 382), (382, 362), (362, 398), (398, 384), (384, 385), (385, 386),
    (386, 387), (387, 388), (388, 466), (466, 263),
    # Face oval
    (10, 338), (338, 297), (297, 332), (332, 284), (284, 251), (251, 389),
    (389, 356), (356, 454), (454, 323), (323, 361), (361, 340), (340, 346),
    (346, 347), (347, 348), (348, 349), (349, 350), (350, 451), (451, 452),
    (452, 453), (453, 464), (464, 435), (435, 410), (410, 287), (287, 273),
    (273, 335), (335, 406), (406, 313), (313, 18), (18, 83), (83, 182),
    (182, 106), (106, 43), (43, 57), (57, 186), (186, 92), (92, 165),
    (165, 167), (167, 164), (164, 393), (393, 391), (391, 322), (322, 270),
    (270, 269), (269, 267), (267, 271), (271, 272), (272, 278), (278, 279),
    (279, 280), (280, 411), (411, 415), (415, 308), (308, 310), (310, 311),
    (311, 312), (312, 13), (13, 82), (82, 81), (81, 80), (80, 78), (78, 62),
    (62, 76), (76, 61), (61, 77), (77, 90), (90, 180), (180, 85), (85, 16),
    (16, 15), (15, 14), (14, 317), (317, 402), (402, 318), (318, 324),
    (324, 308), (308, 415), (415, 411), (411, 280), (280, 279), (279, 278),
    (278, 272), (272, 271), (271, 267), (267, 269), (269, 270), (270, 322),
    (322, 391), (391, 393), (393, 164), (164, 167), (167, 165), (165, 92),
    (92, 186), (186, 57), (57, 43), (43, 106), (106, 182), (182, 83),
    (83, 18), (18, 313), (313, 406), (406, 335), (335, 273), (273, 287),
    (287, 410), (410, 435), (435, 464), (464, 453), (453, 452), (452, 451),
    (451, 350), (350, 349), (349, 348), (348, 347), (347, 346), (346, 340),
    (340, 361), (361, 323), (323, 454), (454, 356), (356, 389), (389, 251),
    (251, 284), (284, 332), (332, 297), (297, 338), (338, 10), (10, 151),
    (151, 337), (337, 299), (299, 333), (333, 298), (298, 301), (301, 368),
    (368, 264), (264, 447), (447, 366), (366, 401), (401, 371), (371, 266),
    (266, 425), (425, 426), (426, 436), (436, 410)
]

class FaceLandmarkDetectorW8A8:
    def __init__(self, model_path='face_landmark_w8a8.tflite',
                 confidence_threshold=0.5,
                 num_threads=4):
        """
        Initialize Face Landmark Detector with w8a8 quantized model
        
        Args:
            model_path: Path to the w8a8 TFLite model
            confidence_threshold: Minimum confidence for face detection
            num_threads: Number of CPU threads
        """
        self.confidence_threshold = confidence_threshold
        
        if not os.path.exists(model_path):
            print(f"Error: Model file not found: {model_path}")
            print("Download with:")
            print(f"wget https://huggingface.co/qualcomm/MediaPipe-Face-Detection/resolve/main/MediaPipe-Face-Detection_FaceLandmarkDetector_w8a8.tflite -O {model_path}")
            sys.exit(1)
        
        print(f"Loading model: {model_path}")
        
        # Initialize interpreter with optional num_threads support
        try:
            # Try the full constructor with num_threads (newer versions)
            self.interpreter = Interpreter(model_path=model_path, num_threads=num_threads)
            print(f"Interpreter initialized with {num_threads} threads")
        except TypeError:
            # Fall back to basic constructor
            self.interpreter = Interpreter(model_path=model_path)
            print("Interpreter initialized (single-threaded mode)")
            
            # Try to set threads separately if method exists
            if hasattr(self.interpreter, 'set_num_threads'):
                try:
                    self.interpreter.set_num_threads(num_threads)
                    print(f"Set to use {num_threads} threads")
                except Exception as e:
                    print(f"Note: Could not set thread count: {e}")
        
        # Allocate tensors
        self.interpreter.allocate_tensors()
        
        # Get input and output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Model input specifications
        self.input_shape = self.input_details[0]['shape']
        self.input_height = self.input_shape[1]
        self.input_width = self.input_shape[2]
        self.input_dtype = self.input_details[0]['dtype']
        
        # Check quantization parameters
        self.is_quantized = self.input_dtype == np.uint8
        if self.is_quantized:
            quant_params = self.input_details[0].get('quantization', (1.0, 0))
            self.input_scale = quant_params[0] if len(quant_params) > 0 else 1.0
            self.input_zero_point = quant_params[1] if len(quant_params) > 1 else 0
            print(f"Quantized model - scale: {self.input_scale}, zero_point: {self.input_zero_point}")
        
        print(f"Model loaded! Input: {self.input_shape}, dtype: {self.input_dtype}")
        print(f"Number of outputs: {len(self.output_details)}")
        
        # Print output details for debugging
        for i, output in enumerate(self.output_details):
            print(f"Output {i}: shape={output['shape']}, dtype={output['dtype']}")
    
    def preprocess_image(self, image):
        """
        Preprocess image for model input
        """
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize to model input size (typically 192x192 for face landmarks)
        resized = cv2.resize(rgb_image, (self.input_width, self.input_height))
        
        # Prepare input based on model requirements
        if self.is_quantized:
            # Quantized model expects uint8 input
            input_data = resized.astype(np.uint8)
        else:
            # Float model expects normalized input
            input_data = resized.astype(np.float32) / 255.0
        
        # Add batch dimension
        input_data = np.expand_dims(input_data, axis=0)
        
        return input_data
    
    def detect_landmarks(self, image):
        """
        Detect face landmarks in image
        
        Returns:
            Dictionary with 'landmarks' (468 points) and 'confidence'
        """
        h, w = image.shape[:2]
        
        # Preprocess
        input_data = self.preprocess_image(image)
        
        # Set input tensor
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        
        # Run inference
        self.interpreter.invoke()
        
        # Get outputs
        outputs = []
        for output_detail in self.output_details:
            output_data = self.interpreter.get_tensor(output_detail['index'])
            outputs.append(output_data)
        
        # Parse outputs
        # MediaPipe Face Landmark model typically outputs:
        # - 468 3D landmarks (x, y, z coordinates)
        # - Face detection confidence/presence score
        
        result = {'landmarks': None, 'confidence': 0.0}
        
        # The main output should be landmarks
        if len(outputs) > 0:
            landmarks_output = outputs[0]
            
            # Remove batch dimension
            if len(landmarks_output.shape) > 2:
                landmarks_output = landmarks_output[0]
            
            # Check shape and reshape if necessary
            if landmarks_output.shape[0] == 468 and landmarks_output.shape[1] >= 2:
                # Already in correct shape [468, 3]
                landmarks = landmarks_output
            elif landmarks_output.shape[0] == 1404:
                # Flattened format [1404] = 468 * 3
                landmarks = landmarks_output.reshape(468, 3)
            else:
                # Try to reshape based on total elements
                total_elements = np.prod(landmarks_output.shape)
                if total_elements == 1404:
                    landmarks = landmarks_output.flatten().reshape(468, 3)
                elif total_elements == 936:  # 468 * 2 (only x, y)
                    landmarks = landmarks_output.flatten().reshape(468, 2)
                    # Add dummy z coordinate
                    landmarks = np.concatenate([landmarks, np.zeros((468, 1))], axis=1)
                else:
                    print(f"Unexpected landmark shape: {landmarks_output.shape}")
                    return result
            
            # Scale landmarks to image coordinates
            scaled_landmarks = []
            for landmark in landmarks:
                x = landmark[0] * w
                y = landmark[1] * h
                z = landmark[2] if len(landmark) > 2 else 0
                scaled_landmarks.append([x, y, z])
            
            result['landmarks'] = np.array(scaled_landmarks)
            
            # Get confidence if available
            if len(outputs) > 1:
                confidence = outputs[1]
                if isinstance(confidence, np.ndarray):
                    confidence = float(confidence.flatten()[0])
                result['confidence'] = confidence
            else:
                # If no separate confidence, assume detection if landmarks are valid
                result['confidence'] = 1.0 if result['landmarks'] is not None else 0.0
        
        return result
    
    def draw_landmarks(self, image, landmarks, draw_connections=True, draw_points=True):
        """
        Draw face landmarks on image
        
        Args:
            image: Input image
            landmarks: 468 face landmarks
            draw_connections: Draw face mesh connections
            draw_points: Draw individual landmark points
        """
        if landmarks is None:
            return image
        
        output_image = image.copy()
        
        # Draw connections (face mesh)
        if draw_connections:
            for connection in FACE_CONNECTIONS:
                start_idx, end_idx = connection
                if start_idx < len(landmarks) and end_idx < len(landmarks):
                    start_point = (int(landmarks[start_idx][0]), int(landmarks[start_idx][1]))
                    end_point = (int(landmarks[end_idx][0]), int(landmarks[end_idx][1]))
                    cv2.line(output_image, start_point, end_point, (0, 255, 0), 1)
        
        # Draw landmark points
        if draw_points:
            for i, landmark in enumerate(landmarks):
                x, y = int(landmark[0]), int(landmark[1])
                
                # Different colors for different face regions
                if i < 17:  # Jaw line
                    color = (255, 0, 0)
                elif i < 27:  # Right eyebrow
                    color = (255, 128, 0)
                elif i < 36:  # Left eyebrow
                    color = (255, 255, 0)
                elif i < 48:  # Right eye
                    color = (0, 255, 0)
                elif i < 68:  # Left eye
                    color = (0, 255, 255)
                elif i < 86:  # Nose
                    color = (0, 0, 255)
                elif i < 114:  # Mouth
                    color = (255, 0, 255)
                else:  # Rest of face mesh
                    color = (128, 128, 128)
                
                cv2.circle(output_image, (x, y), 1, color, -1)
        
        return output_image

class CameraStream:
    """Optimized camera streaming with threading"""
    def __init__(self, src=0, resolution=(320, 240)):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.stream.set(cv2.CAP_PROP_FPS, 10)
        
        self.grabbed, self.frame = self.stream.read()
        self.running = True
    
    def start(self):
        Thread(target=self.update, daemon=True).start()
        return self
    
    def update(self):
        while self.running:
            self.grabbed, self.frame = self.stream.read()
    
    def read(self):
        return self.frame
    
    def stop(self):
        self.running = False
        self.stream.release()

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Face Landmark Detection on Pi Zero 2W')
    parser.add_argument('--model', type=str, default='face_landmark_w8a8.tflite',
                       help='Path to TFLite model')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera index')
    parser.add_argument('--width', type=int, default=320,
                       help='Frame width')
    parser.add_argument('--height', type=int, default=240,
                       help='Frame height')
    parser.add_argument('--skip-frames', type=int, default=3,
                       help='Process every Nth frame')
    parser.add_argument('--no-connections', action='store_true',
                       help='Disable drawing connections')
    parser.add_argument('--save-video', type=str, default=None,
                       help='Save output video to file')
    args = parser.parse_args()
    
    print("=" * 60)
    print("MediaPipe Face Landmark Detector (w8a8) on Pi Zero 2W")
    print("=" * 60)
    
    # Initialize detector
    detector = FaceLandmarkDetectorW8A8(model_path=args.model)
    
    # Initialize camera
    print(f"Starting camera ({args.width}x{args.height})...")
    camera = CameraStream(src=args.camera, resolution=(args.width, args.height)).start()
    time.sleep(2)  # Let camera warm up
    
    # Video writer (optional)
    video_writer = None
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(args.save_video, fourcc, 10.0, (args.width, args.height))
    
    # Performance tracking
    fps_window = []
    frame_count = 0
    landmarks_detected = 0
    
    print("\nControls:")
    print("  'q' - Quit")
    print("  's' - Save screenshot")
    print("  'c' - Toggle connections")
    print("  'p' - Toggle points")
    print("-" * 60)
    
    draw_connections = not args.no_connections
    draw_points = True
    
    try:
        while True:
            frame = camera.read()
            if frame is None:
                continue
            
            frame_count += 1
            
            # Skip frames for performance
            if frame_count % args.skip_frames != 0:
                cv2.imshow('Face Landmarks', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue
            
            # Process frame
            start_time = time.time()
            result = detector.detect_landmarks(frame)
            inference_time = time.time() - start_time
            
            # Draw results
            if result['landmarks'] is not None:
                frame = detector.draw_landmarks(frame, result['landmarks'], 
                                               draw_connections, draw_points)
                landmarks_detected += 1
            
            # Update FPS
            fps_window.append(inference_time)
            if len(fps_window) > 10:
                fps_window.pop(0)
            
            # Display stats
            if len(fps_window) > 0:
                avg_time = sum(fps_window) / len(fps_window)
                fps = 1.0 / avg_time if avg_time > 0 else 0
                
                # Add text overlay
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, f"Inference: {inference_time*1000:.1f}ms", (10, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                if result['landmarks'] is not None:
                    cv2.putText(frame, f"Landmarks: 468", (10, 75),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Save video frame
            if video_writer:
                video_writer.write(frame)
            
            # Display
            cv2.imshow('Face Landmarks', frame)
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f'landmarks_{int(time.time())}.jpg'
                cv2.imwrite(filename, frame)
                print(f"Screenshot saved: {filename}")
            elif key == ord('c'):
                draw_connections = not draw_connections
                print(f"Connections: {'ON' if draw_connections else 'OFF'}")
            elif key == ord('p'):
                draw_points = not draw_points
                print(f"Points: {'ON' if draw_points else 'OFF'}")
    
    except KeyboardInterrupt:
        print("\nInterrupted!")
    finally:
        print("\nCleaning up...")
        camera.stop()
        if video_writer:
            video_writer.release()
        cv2.destroyAllWindows()
        
        # Print statistics
        print("\nStatistics:")
        print(f"  Total frames: {frame_count}")
        print(f"  Frames with landmarks: {landmarks_detected}")
        if len(fps_window) > 0:
            print(f"  Average FPS: {1.0/(sum(fps_window)/len(fps_window)):.2f}")

if __name__ == "__main__":
    main()
