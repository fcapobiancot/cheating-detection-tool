import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, Label, Button, Frame, StringVar, ttk
from PIL import Image, ImageTk
import tensorflow as tf
import threading
import time
from datetime import timedelta
import tempfile
import subprocess
import wave
import array
import math
from moviepy.editor import VideoFileClip

class CheatDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Online Exam Cheating Detection")
        self.root.geometry("900x750")
        self.root.configure(bg="#f0f0f0")
        
        # State variables
        self.video_path = None
        self.is_processing = False
        self.processing_thread = None
        self.cap = None
        
        # Model variables
        self.video_interpreter = None
        self.video_input_details = None
        self.video_output_details = None
        self.video_class_labels = []
        
        self.audio_interpreter = None
        self.audio_input_details = None
        self.audio_output_details = None
        self.audio_class_labels = []
        
        # Load models and labels
        self.load_models()
        
        # Create UI elements
        self.create_ui()
    
    def load_models(self):
        try:
            # Load video model and labels
            with open("labels.txt", "r") as label_file:
                for line in label_file:
                    self.video_class_labels.append(line.strip().split(" ", 1)[1])
            
            self.video_interpreter = tf.lite.Interpreter(model_path="model.tflite")
            self.video_interpreter.allocate_tensors()
            self.video_input_details = self.video_interpreter.get_input_details()
            self.video_output_details = self.video_interpreter.get_output_details()
            
            print("Video model loaded successfully")
            print(f"Input shape: {self.video_input_details[0]['shape']}")
            print(f"Video classes: {self.video_class_labels}")
            
            # Load audio model and labels
            with open("audio_labels.txt", "r") as label_file:
                for line in label_file:
                    self.audio_class_labels.append(line.strip().split(" ", 1)[1])
            
            self.audio_interpreter = tf.lite.Interpreter(model_path="audio_model.tflite")
            self.audio_interpreter.allocate_tensors()
            self.audio_input_details = self.audio_interpreter.get_input_details()
            self.audio_output_details = self.audio_interpreter.get_output_details()
            
            print("Audio model loaded successfully")
            print(f"Input shape: {self.audio_input_details[0]['shape']}")
            print(f"Audio classes: {self.audio_class_labels}")
            
        except Exception as e:
            print(f"Error loading models: {e}")
    
    def create_ui(self):
        # Header
        header_frame = Frame(self.root, bg="#3498db", padx=10, pady=10)
        header_frame.pack(fill=tk.X)
        
        header_label = Label(header_frame, text="Online Exam Cheating Detection", 
                            font=("Arial", 18, "bold"), fg="white", bg="#3498db")
        header_label.pack()
        
        # Main content area
        content_frame = Frame(self.root, bg="#f0f0f0", padx=20, pady=20)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Video selection section
        select_frame = Frame(content_frame, bg="#f0f0f0", pady=10)
        select_frame.pack(fill=tk.X)
        
        self.path_var = StringVar()
        self.path_var.set("No video selected")
        
        select_btn = Button(select_frame, text="Select Video", command=self.select_video, 
                           bg="#2980b9", fg="white", font=("Arial", 12), padx=10, pady=5)
        select_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        path_label = Label(select_frame, textvariable=self.path_var, bg="#f0f0f0", 
                          font=("Arial", 10), anchor="w")
        path_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Video preview and results area
        display_frame = Frame(content_frame, bg="#f0f0f0")
        display_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Video preview
        self.preview_frame = Frame(display_frame, bg="black", width=640, height=480)
        self.preview_frame.pack(side=tk.LEFT, padx=(0, 10))
        self.preview_frame.pack_propagate(False)
        
        self.video_label = Label(self.preview_frame, bg="black")
        self.video_label.pack(fill=tk.BOTH, expand=True)
        
        # Results section
        results_frame = Frame(display_frame, bg="#f0f0f0", width=200)
        results_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        Label(results_frame, text="Detection Results", font=("Arial", 14, "bold"), 
              bg="#f0f0f0").pack(pady=(0, 10))
        
        self.progress_var = StringVar()
        self.progress_var.set("Progress: 0%")
        self.progress_label = Label(results_frame, textvariable=self.progress_var, 
                                   bg="#f0f0f0", font=("Arial", 10))
        self.progress_label.pack(anchor="w", pady=5)
        
        self.progress_bar = ttk.Progressbar(results_frame, orient=tk.HORIZONTAL, length=200, mode='determinate')
        self.progress_bar.pack(fill=tk.X, pady=(0, 10))
        
        # Stats frame for video
        video_stats_frame = Frame(results_frame, bg="#f0f0f0")
        video_stats_frame.pack(fill=tk.X, pady=10)
        
        Label(video_stats_frame, text="Video Analysis:", bg="#f0f0f0", font=("Arial", 12, "bold"), 
              anchor="w").pack(fill=tk.X, pady=2)
        
        # Create variables and labels for video class percentages
        self.normal_var = StringVar(value="Normal: 0%")
        self.looking_away_var = StringVar(value="Looking Away: 0%")
        self.no_one_var = StringVar(value="No One in Focus: 0%")
        
        Label(video_stats_frame, textvariable=self.normal_var, bg="#f0f0f0", font=("Arial", 12), 
              anchor="w").pack(fill=tk.X, pady=2)
        Label(video_stats_frame, textvariable=self.looking_away_var, bg="#f0f0f0", font=("Arial", 12), 
              anchor="w").pack(fill=tk.X, pady=2)
        Label(video_stats_frame, textvariable=self.no_one_var, bg="#f0f0f0", font=("Arial", 12), 
              anchor="w").pack(fill=tk.X, pady=2)
        
        # Stats frame for audio
        audio_stats_frame = Frame(results_frame, bg="#f0f0f0")
        audio_stats_frame.pack(fill=tk.X, pady=10)
        
        Label(audio_stats_frame, text="Audio Analysis:", bg="#f0f0f0", font=("Arial", 12, "bold"), 
              anchor="w").pack(fill=tk.X, pady=2)
        
        # Create variables and labels for audio class percentages
        self.background_var = StringVar(value="Background Noise: 0%")
        self.talking_var = StringVar(value="Talking: 0%")
        
        Label(audio_stats_frame, textvariable=self.background_var, bg="#f0f0f0", font=("Arial", 12), 
              anchor="w").pack(fill=tk.X, pady=2)
        Label(audio_stats_frame, textvariable=self.talking_var, bg="#f0f0f0", font=("Arial", 12), 
              anchor="w").pack(fill=tk.X, pady=2)
        
        # Final result
        result_frame = Frame(results_frame, bg="#f0f0f0", pady=10)
        result_frame.pack(fill=tk.X)
        
        self.result_var = StringVar(value="Result: Waiting for analysis")
        self.result_label = Label(result_frame, textvariable=self.result_var, bg="#f0f0f0", 
                                 font=("Arial", 14, "bold"), pady=10)
        self.result_label.pack(fill=tk.X)
        
        # Detailed result
        self.detail_var = StringVar(value="")
        self.detail_label = Label(result_frame, textvariable=self.detail_var, bg="#f0f0f0", 
                                font=("Arial", 10), wraplength=200)
        self.detail_label.pack(fill=tk.X)
        
        # Control buttons
        control_frame = Frame(content_frame, bg="#f0f0f0", pady=10)
        control_frame.pack(fill=tk.X)
        
        self.analyze_btn = Button(control_frame, text="Analyze Video", command=self.start_analysis, 
                                 bg="#27ae60", fg="white", font=("Arial", 12), padx=10, pady=5)
        self.analyze_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        self.stop_btn = Button(control_frame, text="Stop Analysis", command=self.stop_analysis, 
                              bg="#e74c3c", fg="white", font=("Arial", 12), padx=10, pady=5, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT)
        
        # Status bar
        self.status_var = StringVar()
        self.status_var.set("Ready")
        status_bar = Label(self.root, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def select_video(self):
        file_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")]
        )
        
        if file_path:
            self.video_path = file_path
            self.path_var.set(os.path.basename(file_path))
            self.status_var.set(f"Selected video: {os.path.basename(file_path)}")
            
            # Show the first frame as preview
            self.show_video_preview()
    
    def show_video_preview(self):
        if self.video_path:
            cap = cv2.VideoCapture(self.video_path)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = self.resize_frame(frame)
                img = Image.fromarray(frame)
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)
            cap.release()
    
    def resize_frame(self, frame):
        target_width = 640
        target_height = 480
        
        h, w = frame.shape[:2]
        
        # Calculate scaling factor to maintain aspect ratio
        if h > w:
            scale = target_height / h
            new_h = target_height
            new_w = int(w * scale)
        else:
            scale = target_width / w
            new_w = target_width
            new_h = int(h * scale)
        
        resized = cv2.resize(frame, (new_w, new_h))
        
        # Create a black canvas of target size
        canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        
        # Calculate position to center the image
        y_offset = (target_height - new_h) // 2
        x_offset = (target_width - new_w) // 2
        
        # Place the resized image on the canvas
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        return canvas
    
    def preprocess_video_frame(self, frame):
        # Get model input shape
        input_shape = self.video_input_details[0]['shape'][1:3]  # [height, width]
        
        # Resize to match model input
        resized = cv2.resize(frame, (input_shape[1], input_shape[0]))
        
        # Normalize pixel values to [0, 1]
        normalized = resized.astype(np.float32) / 255.0
        
        # Add batch dimension
        input_data = np.expand_dims(normalized, axis=0)
        
        return input_data
    
    def predict_video_frame(self, frame):
        # Convert BGR to RGB (if needed)
        if frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Preprocess the image
        input_data = self.preprocess_video_frame(frame)
        
        # Set input tensor
        self.video_interpreter.set_tensor(self.video_input_details[0]['index'], input_data)
        
        # Run inference
        self.video_interpreter.invoke()
        
        # Get output tensor
        output_data = self.video_interpreter.get_tensor(self.video_output_details[0]['index'])
        
        # Get predicted class
        predicted_class = np.argmax(output_data)
        confidence = output_data[0][predicted_class]
        
        return predicted_class, confidence
    
    def extract_audio_from_video(self, video_path):
        """Extract audio from video file and return path to temporary audio file"""
        try:
            self.status_var.set("Extracting audio from video...")
            
            # Create a temporary WAV file
            audio_path = os.path.join(tempfile.gettempdir(), "temp_audio.wav")
            
            # Extract audio using moviepy
            video = VideoFileClip(video_path)
            video.audio.write_audiofile(audio_path, logger=None)
            
            return audio_path
        except Exception as e:
            print(f"Error extracting audio: {e}")
            return None
    
    def analyze_audio(self, audio_path):
        """Analyze audio using TFLite model and return class counts"""
        try:
            if not os.path.exists(audio_path):
                return {0: 1, 1: 0}  # Default to background noise if no audio
                
            # Load WAV file
            with wave.open(audio_path, 'rb') as wav_file:
                n_channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                frame_rate = wav_file.getframerate()
                n_frames = wav_file.getnframes()
                
                # Read all frames
                frames = wav_file.readframes(n_frames)
                
                # Convert to numpy array
                if sample_width == 1:
                    dtype = np.uint8
                elif sample_width == 2:
                    dtype = np.int16
                elif sample_width == 4:
                    dtype = np.int32
                else:
                    raise ValueError("Unsupported sample width")
                
                # Convert bytes to numpy array
                audio_data = np.frombuffer(frames, dtype=dtype)
                
                # Only use one channel if stereo
                if n_channels == 2:
                    audio_data = audio_data[::2]  # Take every other sample (left channel)
            
            # Get model input shape
            input_shape = self.audio_input_details[0]['shape']
            
            # Define segment length and hop length (overlap)
            segment_length = input_shape[1]  # Model's expected input length
            hop_length = segment_length // 2  # 50% overlap
            
            # Calculate number of segments
            num_segments = (len(audio_data) - segment_length) // hop_length + 1
            
            # Initialize class counts
            class_counts = {0: 0, 1: 0}  # Background, Talking
            
            # Process audio in segments
            for i in range(min(100, num_segments)):  # Limit to 100 segments for speed
                start = i * hop_length
                end = start + segment_length
                
                if end > len(audio_data):
                    break
                    
                segment = audio_data[start:end]
                
                # Normalize segment to [-1, 1]
                if dtype == np.uint8:
                    segment = (segment.astype(np.float32) - 128) / 128.0
                else:
                    segment = segment.astype(np.float32) / (2**(8 * sample_width - 1))
                
                # Reshape to match model input
                if len(input_shape) == 2:  # [batch_size, input_length]
                    input_data = np.expand_dims(segment, axis=0)
                elif len(input_shape) == 3:  # [batch_size, input_length, 1]
                    input_data = np.expand_dims(np.expand_dims(segment, axis=0), axis=2)
                
                # Resize if needed
                if len(segment) < segment_length:
                    # Pad with zeros
                    padding = np.zeros(segment_length - len(segment))
                    segment = np.concatenate([segment, padding])
                elif len(segment) > segment_length:
                    # Truncate
                    segment = segment[:segment_length]
                
                # Ensure correct shape
                input_data = np.reshape(segment, input_shape)
                
                # Set input tensor
                self.audio_interpreter.set_tensor(self.audio_input_details[0]['index'], input_data)
                
                # Run inference
                self.audio_interpreter.invoke()
                
                # Get output tensor
                output_data = self.audio_interpreter.get_tensor(self.audio_output_details[0]['index'])
                
                # Get predicted class
                predicted_class = np.argmax(output_data)
                class_counts[predicted_class] += 1
            
            return class_counts
            
        except Exception as e:
            print(f"Error analyzing audio: {e}")
            return {0: 1, 1: 0}  # Default to background noise on error
    
    def start_analysis(self):
        if not self.video_path:
            self.status_var.set("Error: No video selected")
            return
        
        if not self.video_interpreter or not self.audio_interpreter:
            self.status_var.set("Error: Models not loaded properly")
            return
        
        if self.is_processing:
            return
        
        # Reset UI
        self.normal_var.set("Normal: 0%")
        self.looking_away_var.set("Looking Away: 0%")
        self.no_one_var.set("No One in Focus: 0%")
        self.background_var.set("Background Noise: 0%")
        self.talking_var.set("Talking: 0%")
        self.result_var.set("Result: Analyzing...")
        self.detail_var.set("")
        self.progress_bar['value'] = 0
        self.progress_var.set("Progress: 0%")
        
        # Update UI state
        self.is_processing = True
        self.analyze_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        
        # Start processing in a separate thread
        self.processing_thread = threading.Thread(target=self.process_video_and_audio)
        self.processing_thread.daemon = True
        self.processing_thread.start()
    
    def stop_analysis(self):
        if self.is_processing:
            self.is_processing = False
            self.status_var.set("Analysis stopped by user")
            self.result_var.set("Result: Stopped")
            self.analyze_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)
    
    def process_video_and_audio(self):
        try:
            self.status_var.set("Analyzing video and audio...")
            
            # Extract audio from video
            audio_path = self.extract_audio_from_video(self.video_path)
            
            # Open video
            cap = cv2.VideoCapture(self.video_path)
            self.cap = cap
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps
            
            # Initialize variables for counting video frames
            video_class_counts = {0: 0, 1: 0, 2: 0}  # Normal, Looking Away, No One
            processed_frames = 0
            
            # Process every nth frame to speed up analysis
            frame_step = 5
            sampled_frame_count = frame_count // frame_step
            if sampled_frame_count == 0:
                sampled_frame_count = 1
            
            # Loop through video frames
            frame_idx = 0
            while cap.isOpened() and self.is_processing:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process every nth frame
                if frame_idx % frame_step == 0:
                    # Predict frame class
                    predicted_class, confidence = self.predict_video_frame(frame)
                    video_class_counts[predicted_class] += 1
                    
                    # Update progress
                    processed_frames += 1
                    progress = (processed_frames / sampled_frame_count) * 100
                    
                    # Update UI (in main thread)
                    self.root.after(0, self.update_video_progress, progress, frame, video_class_counts, sampled_frame_count)
                
                frame_idx += 1
            
            # Process audio
            audio_class_counts = self.analyze_audio(audio_path)
            
            # Final results update
            if self.is_processing:  # Only if not stopped by user
                self.root.after(0, self.update_final_results, video_class_counts, audio_class_counts, sampled_frame_count)
            
            # Clean up
            cap.release()
            self.cap = None
            
            # Remove temporary audio file
            if audio_path and os.path.exists(audio_path):
                os.remove(audio_path)
            
        except Exception as e:
            self.root.after(0, self.update_error, str(e))
    
    def update_video_progress(self, progress, frame, video_class_counts, total_frames):
        # Update progress bar and label
        self.progress_bar['value'] = progress
        self.progress_var.set(f"Progress: {progress:.1f}%")
        
        # Update video class percentages
        normal_pct = (video_class_counts[0] / total_frames) * 100 if total_frames > 0 else 0
        looking_away_pct = (video_class_counts[1] / total_frames) * 100 if total_frames > 0 else 0
        no_one_pct = (video_class_counts[2] / total_frames) * 100 if total_frames > 0 else 0
        
        self.normal_var.set(f"Normal: {normal_pct:.1f}%")
        self.looking_away_var.set(f"Looking Away: {looking_away_pct:.1f}%")
        self.no_one_var.set(f"No One in Focus: {no_one_pct:.1f}%")
        
        # Update video preview
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = self.resize_frame(frame_rgb)
        img = Image.fromarray(frame_resized)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)
        
        # Update status
        self.status_var.set(f"Analyzing video: {progress:.1f}% complete")
    
    def update_final_results(self, video_class_counts, audio_class_counts, total_frames):
        # Calculate final video percentages
        normal_pct = (video_class_counts[0] / total_frames) * 100 if total_frames > 0 else 0
        looking_away_pct = (video_class_counts[1] / total_frames) * 100 if total_frames > 0 else 0
        no_one_pct = (video_class_counts[2] / total_frames) * 100 if total_frames > 0 else 0
        
        # Update UI with final video results
        self.normal_var.set(f"Normal: {normal_pct:.1f}%")
        self.looking_away_var.set(f"Looking Away: {looking_away_pct:.1f}%")
        self.no_one_var.set(f"No One in Focus: {no_one_pct:.1f}%")
        
        # Calculate audio percentages
        total_audio_segments = sum(audio_class_counts.values())
        background_pct = (audio_class_counts[0] / total_audio_segments) * 100 if total_audio_segments > 0 else 0
        talking_pct = (audio_class_counts[1] / total_audio_segments) * 100 if total_audio_segments > 0 else 0
        
        # Update UI with audio results
        self.background_var.set(f"Background Noise: {background_pct:.1f}%")
        self.talking_var.set(f"Talking: {talking_pct:.1f}%")
        
        # Determine if cheating was detected based on both video and audio
        is_video_cheating = normal_pct < 70
        is_audio_cheating = talking_pct > 35
        
        if is_video_cheating and is_audio_cheating:
            result = "SUSPICIOUS - Cheating detected (Video & Audio)"
            result_color = "#e74c3c"  # Red
            details = "Video shows suspicious behavior AND excessive talking detected."
        elif is_video_cheating:
            result = "SUSPICIOUS - Cheating detected (Video)"
            result_color = "#e74c3c"  # Red
            details = "Video shows suspicious behavior: looking away or absence detected more than 30% of the time."
        elif is_audio_cheating:
            result = "SUSPICIOUS - Cheating detected (Audio)"
            result_color = "#e74c3c"  # Red
            details = "Excessive talking detected above the 35% threshold."
        else:
            result = "NORMAL - No cheating detected"
            result_color = "#27ae60"  # Green
            details = "Both video and audio analysis show normal exam behavior."
        
        self.result_var.set(f"Result: {result}")
        self.result_label.config(fg=result_color)
        self.detail_var.set(details)
        
        # Complete progress bar
        self.progress_bar['value'] = 100
        self.progress_var.set("Progress: 100%")
        
        # Update status
        self.status_var.set("Analysis completed")
        
        # Reset UI state
        self.is_processing = False
        self.analyze_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
    
    def update_error(self, error_msg):
        self.status_var.set(f"Error: {error_msg}")
        self.result_var.set("Result: Error during analysis")
        
        # Reset UI state
        self.is_processing = False
        self.analyze_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)

if __name__ == "__main__":
    root = tk.Tk()
    app = CheatDetectionApp(root)
    root.mainloop()