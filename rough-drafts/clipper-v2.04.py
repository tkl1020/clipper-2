# Video Player + Enhanced AI-Based Highlight Detection
# ==========================================

# These are 'import' statements - they bring in code libraries that other developers have written
# so we can use their functionality without having to write everything from scratch
import sys                # Provides access to system-specific parameters and functions
import os                 # Allows interaction with the operating system (files, directories, etc.)
import whisper            # An AI library for transcribing speech from audio to text
import time               # Provides various time-related functions
import concurrent.futures
import gc                 # Garbage collection for memory management
# psutil is optional for system resource monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
import queue              # For work queue management
import threading          # For thread management
from transformers import pipeline  # AI-based text processing library

# PyQt5 imports - these are for creating the graphical user interface (GUI)
# QApplication is the core of any PyQt application
# Various widgets like buttons, text boxes, layouts are imported to build the interface
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QTextEdit, QVBoxLayout, QHBoxLayout,
    QFileDialog, QLabel, QProgressBar, QSlider, QStyle, QMessageBox,
    QLineEdit, QFrame
)
# More PyQt imports for handling core functionality and timing
from PyQt5.QtCore import Qt, QUrl, QTimer, QDir, QThread, pyqtSignal
# PyQt imports for text formatting and display
from PyQt5.QtGui import QTextCursor, QFont, QColor

# PyQt libraries for multimedia handling (playing videos)
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
# MoviePy library for video editing capabilities
from moviepy.editor import VideoFileClip

# Try to import pydub for silence detection - not required but enhances processing
try:
    from pydub import AudioSegment
    from pydub.silence import detect_silence
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False

# Import libraries for audio visualizer
from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QPainter, QPen
import numpy as np
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

# Load an AI text classification model that will detect emotions in text
# This will be used to find potentially interesting/funny moments in videos
classifier = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion", top_k=2)

# Audio Visualizer Class
class WaveformWidget(QWidget):
    def __init__(self, samples, sample_rate, duration, parent=None):
        super().__init__(parent)
        self.samples = samples
        self.sample_rate = sample_rate
        self.duration = duration
        self.position = 0
        self.setMinimumHeight(100)
        
        # Timer for animation/playback position updates
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update)
        self.timer.start(50)  # Update every 50ms
    
    def set_position(self, position):
        """Sets current playback position in seconds"""
        self.position = position
        self.update()
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Background
        painter.fillRect(event.rect(), QColor(40, 40, 40))
        
        # Get widget dimensions
        width = self.width()
        height = self.height()
        
        # Calculate how many samples to skip to fit the view
        samples_per_pixel = max(1, len(self.samples) // width)
        
        # Draw waveform
        pen = QPen(QColor(0, 200, 100))
        pen.setWidth(1)
        painter.setPen(pen)
        
        # Calculate and draw waveform segments
        middle_y = height // 2
        scale_factor = height * 0.4
        
        # Pre-calculate values for performance
        x_values = range(0, width)
        
        for i in x_values:
            start_idx = i * samples_per_pixel
            end_idx = min(start_idx + samples_per_pixel, len(self.samples))
            
            if start_idx < len(self.samples):
                segment = self.samples[start_idx:end_idx]
                if len(segment) > 0:
                    segment_max = np.max(segment)
                    segment_min = np.min(segment)
                    
                    # Scale amplitude to widget height
                    y_max = middle_y - int(segment_max * scale_factor)
                    y_min = middle_y - int(segment_min * scale_factor)
                    
                    painter.drawLine(i, y_max, i, y_min)
        
        # Draw playback position line
        if self.duration > 0:
            pos_x = int((self.position / self.duration) * width)
            playback_pen = QPen(QColor(255, 100, 0))
            playback_pen.setWidth(2)
            painter.setPen(playback_pen)
            painter.drawLine(pos_x, 0, pos_x, height)

# This class creates a separate thread for running the transcription process
# Using a separate thread prevents the application from freezing while processing
class TranscriptionWorker(QThread):
    progress = pyqtSignal(int)
    live_update = pyqtSignal(str)
    finished = pyqtSignal(list)

    def __init__(self, model, audio_path):
        super().__init__()
        self.model = model
        self.audio_path = audio_path
        
        # Default batch and worker settings if psutil not available
        self.batch_size = 10
        self.max_workers = 2
        
        # Try to use psutil for adaptive resource management if available
        if 'PSUTIL_AVAILABLE' in globals() and PSUTIL_AVAILABLE:
            try:
                # Adaptive batch sizing based on system resources
                cpu_cores = psutil.cpu_count(logical=False) or 2
                mem_available_gb = psutil.virtual_memory().available / (1024 * 1024 * 1024)
                
                # Adjust batch size based on available resources
                if mem_available_gb > 8 and cpu_cores >= 4:
                    self.batch_size = 20
                elif mem_available_gb > 4 and cpu_cores >= 2:
                    self.batch_size = 10
                else:
                    self.batch_size = 5
                
                self.max_workers = max(1, min(cpu_cores - 1, 4))  # Leave one core free
            except Exception:
                # Fall back to defaults if there's an error
                pass
                
        self.use_threading = True  # Enable parallel processing

    def run(self):
        # Get full transcription in one go
        result = self.model.transcribe(
            self.audio_path,
            fp16=False,  # Use fp32 for CPU
            language="en",  # Specify language if known
        )
        
        segments = result['segments']
        total_segments = len(segments)
        
        # Try to detect silence for better segmentation
        silence_timestamps = []
        if PYDUB_AVAILABLE:
            try:
                sound = AudioSegment.from_file(self.audio_path)
                silence_regions = detect_silence(sound, min_silence_len=500, silence_thresh=-40)
                
                # Convert silence regions to timestamps (ms → seconds)
                silence_timestamps = [(start/1000, end/1000) for start, end in silence_regions]
                self.live_update.emit("Located natural breaks in audio for better processing")
            except Exception as e:
                print(f"Silence detection error: {e}")
        
        # Pre-process segments into smaller, uniform chunks
        processed_segments = []
        for segment in segments:
            text = segment['text'].strip()
            start = segment['start']
            end = segment['end']
            
            # Use silence detection for more natural segment breaks if available
            if silence_timestamps:
                # Find silence regions within this segment
                segment_silences = [
                    (s_start, s_end) for s_start, s_end in silence_timestamps 
                    if s_start >= start and s_end <= end
                ]
                
                if segment_silences:
                    # Use silence points as natural breaking points
                    last_point = start
                    for s_start, s_end in segment_silences:
                        if s_start - last_point > 1.0:  # At least 1 second of speech
                            processed_segments.append((last_point, text))
                        last_point = s_end
                    if end - last_point > 1.0:
                        processed_segments.append((last_point, text))
                    continue
            
            # Fall back to time-based chunking if no silences found
            if end - start > 10:
                slice_size = 10  # seconds
                num_slices = int((end - start) // slice_size) + 1
                for slice_idx in range(num_slices):
                    slice_start = start + slice_idx * slice_size
                    slice_end = min(slice_start + slice_size, end)
                    processed_segments.append((slice_start, text))
            else:
                processed_segments.append((start, text))
        
        # Process in batches using a work queue
        work_queue = queue.Queue()
        result_queue = queue.Queue()
        detected_highlights = []
        
        # Fill queue with work
        for segment in processed_segments:
            work_queue.put(segment)
        
        # Worker function for thread pool
        def worker():
            while True:
                try:
                    segment = work_queue.get(timeout=1)
                    if segment is None:  # Sentinel to signal end
                        break
                    result = self._process_segment(segment)
                    if result:
                        result_queue.put(result)
                    work_queue.task_done()
                except queue.Empty:
                    break
                except Exception as e:
                    print(f"Worker error: {e}")
                    work_queue.task_done()
        
        # Add sentinels to stop workers
        for _ in range(self.max_workers):
            work_queue.put(None)
        
        # Create progress tracking variables
        total_items = len(processed_segments)
        completed_items = 0
        
        # Start workers if using threading
        if self.use_threading:
            threads = []
            for _ in range(self.max_workers):
                t = threading.Thread(target=worker)
                t.daemon = True
                t.start()
                threads.append(t)
            
            # Monitor progress while workers are running
            last_update_time = time.time()
            update_interval = 0.5  # seconds
            
            while any(t.is_alive() for t in threads):
                # Calculate approximate progress
                current_size = work_queue.qsize()
                if total_items > 0:
                    completed = total_items - current_size
                    percent_complete = min(100, int((completed / total_items) * 100))
                    
                    # Update progress bar periodically
                    current_time = time.time()
                    if current_time - last_update_time > update_interval:
                        self.progress.emit(percent_complete)
                        last_update_time = current_time
                
                # Process any available results
                highlights_batch = []
                while not result_queue.empty():
                    highlights_batch.append(result_queue.get())
                
                # Send batch updates
                if highlights_batch:
                    highlights_text = "\n".join(f"Potential highlight found: {h[2]}" for h in highlights_batch)
                    self.live_update.emit(highlights_text)
                    detected_highlights.extend(highlights_batch)
                
                # Sleep briefly to prevent high CPU usage in this loop
                time.sleep(0.1)
                
                # Periodically force garbage collection
                if int(current_time) % 5 == 0:
                    gc.collect()
            
            # Wait for all threads to complete
            for t in threads:
                t.join()
        else:
            # Sequential processing fallback
            for i, segment in enumerate(processed_segments):
                result = self._process_segment(segment)
                if result:
                    detected_highlights.append(result)
                    self.live_update.emit(f"Highlight found: {result[2]}")
                
                # Update progress every few items
                if i % 5 == 0:
                    percent_complete = min(100, int((i / total_items) * 100))
                    self.progress.emit(percent_complete)
        
        # Get any remaining results from the queue
        while not result_queue.empty():
            detected_highlights.append(result_queue.get())
        
        # Final progress update
        self.progress.emit(100)
        
        # Force garbage collection before finishing
        gc.collect()
        
        self.finished.emit(detected_highlights)

    def _process_segment(self, segment_data):
        """Process a single segment and return highlight if found"""
        timestamp, text = segment_data
        try:
            # Pre-compute multiple classifications at once
            prediction = classifier(text)
            
            # Get top emotions and scores
            top_label = prediction[0][0]['label']
            top_score = prediction[0][0]['score']
            
            # Multiple emotion categories with customized thresholds
            highlight_emotions = {
                "joy": 0.92,        # Happy moments
                "surprise": 0.92,   # Unexpected moments
                "anger": 0.90,      # Dramatic or intense moments
                "fear": 0.90,       # Suspenseful moments
                "sadness": 0.92,    # Emotional or touching moments
            }
            
            # Get second highest emotion for context
            second_label = prediction[0][1]['label'] if len(prediction[0]) > 1 else None
            second_score = prediction[0][1]['score'] if len(prediction[0]) > 1 else 0
            
            # Check if top emotion meets threshold
            if top_label in highlight_emotions and top_score > highlight_emotions[top_label]:
                # Create highlight with adjusted timestamps based on emotion type
                if top_label in ["surprise", "fear"]:
                    # For surprise or fear, include more lead-up time
                    clip_start = max(0, timestamp - 3)
                    clip_end = timestamp + 7
                elif top_label == "anger":
                    # For anger, include more aftermath
                    clip_start = max(0, timestamp - 1.5)
                    clip_end = timestamp + 10
                else:
                    # Default timing for other emotions
                    clip_start = max(0, timestamp - 2)
                    clip_end = timestamp + 8
                
                # Add emotion label to text
                labeled_text = f"[{top_label.upper()}] {text}"
                
                return (clip_start, clip_end, labeled_text, top_label)
        except Exception as e:
            # Silently ignore errors in emotional processing
            pass
        return None

    @staticmethod
    def optimize_classifier_for_batching(classifier):
        """
        Modify the classifier to accept batches of text
        This is a placeholder - implementation depends on your classifier type
        """
        try:
            return pipeline(
                "text-classification", 
                model=classifier.model,
                tokenizer=classifier.tokenizer,
                batch_size=8,
                truncation=True
            )
        except:
            return classifier  # Default: return original classifier

# This is the main application class that creates the video editor interface
class VideoTranscriberEditor(QWidget):
    # This is the constructor - it sets up everything when the application starts
    def __init__(self):
        super().__init__()  # Initialize the parent class (QWidget)

        # Load the speech recognition model (whisper)
        # "tiny" is the model size - smaller is faster but less accurate
        self.model = whisper.load_model("tiny")
        import torch
        # If a GPU is available, use it for faster processing
        if torch.cuda.is_available():
            self.model = self.model.to("cuda")

        # Initialize variables for the video editor
        self.video_file_path = None      # Path to the video file
        self.video_clip = None           # The actual video content
        self.current_time = 0            # Current playback position in seconds
        self.clip_start_time = None      # Where to start cutting a clip
        self.clip_end_time = None        # Where to end cutting a clip
        self.is_playing = False          # Whether the video is currently playing
        self.is_audio_only = False       # Whether the loaded file is audio-only
        
        # Initialize audio visualizer components
        self.visualizer_container = None # Container for the waveform widget
        self.waveform = None             # The waveform widget itself

        # Initialize variables for transcription
        self.audio_path = None           # Path to the audio file
        self.full_text = ""              # Complete transcription text
        self.pending_segments = []       # Text segments waiting to be displayed
        self.current_typing_text = ""    # Text currently being "typed" onto the screen
        self.current_char_index = 0      # Position in the current text being typed
        # Timer to control the typing animation
        self.typing_timer = QTimer()
        self.typing_timer.timeout.connect(self.type_next_character)

        # List to store detected highlight moments
        self.highlights = []
        
        # Current highlight index when browsing through highlights
        self.current_highlight_index = -1
        
        # Create a frame for the video player
        self.video_frame = QFrame()
        self.video_frame.setFrameShape(QFrame.StyledPanel)
        self.video_frame.setFrameShadow(QFrame.Raised)
        self.video_frame.setStyleSheet("background-color: #1a130a;")  # Darker background for video

        # Call methods to set up the application
        self.setup_window()              # Configure the main window
        self.create_ui_components()      # Create buttons, sliders, etc.
        self.setup_media_player()        # Set up the video player
        self.create_layouts()            # Arrange the UI elements
        self.setup_connections()         # Connect buttons to functions

        self.apply_dark_theme()          # Apply a nice dark color scheme

        # Timer to update the playback position display
        self.update_timer = QTimer(self)
        self.update_timer.setInterval(100)  # Update every 100 milliseconds
        self.update_timer.timeout.connect(self.update_playback_position)

    # Creates an audio waveform visualization for audio files
    def create_audio_visualizer(self, audio_path, widget_width, widget_height):
        """
        Creates an audio waveform visualization for the given audio file.
        """
        if not LIBROSA_AVAILABLE:
            print("Warning: librosa is not installed. Audio visualization will not be available.")
            empty_widget = QWidget()
            empty_widget.setFixedSize(widget_width, widget_height)
            return empty_widget, None
            
        try:
            y, sr = librosa.load(audio_path, sr=None)
            duration = librosa.get_duration(y=y, sr=sr)
            
            # Create container widget and layout
            container = QWidget()
            layout = QVBoxLayout(container)
            layout.setContentsMargins(0, 0, 0, 0)
            
            # Create waveform widget
            waveform = WaveformWidget(y, sr, duration)
            layout.addWidget(waveform)
            
            # Set visualization widget size
            container.setFixedSize(widget_width, widget_height)
            
            return container, waveform
        
        except Exception as e:
            print(f"Error creating audio visualizer: {e}")
            # Return an empty widget in case of error
            empty_widget = QWidget()
            empty_widget.setFixedSize(widget_width, widget_height)
            return empty_widget, None

    # Determines if a file is an audio file based on extension
    def is_audio_file(self, file_path):
        """
        Determines if the given file is an audio file based on extension.
        """
        audio_extensions = ['.mp3', '.wav', '.ogg', '.flac', '.aac', '.m4a']
        _, extension = os.path.splitext(file_path.lower())
        return extension in audio_extensions

    # Set up the main application window properties
    def setup_window(self):
        self.setWindowTitle("Clipper v2.1 - Enhanced AI Highlight Detection")
        self.setGeometry(100, 100, 1200, 800)  # (x, y, width, height)
        self.setMinimumSize(900, 600)          # Minimum allowed window size

    # Set up the video player component
    def setup_media_player(self):
        # Create a widget to display the video
        self.video_widget = QVideoWidget(self)
        self.video_widget.setMinimumHeight(360)
        # Create the media player that will handle the video playback
        self.media_player = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.media_player.setVideoOutput(self.video_widget)
        # Connect media player events to our functions
        self.media_player.stateChanged.connect(self.media_state_changed)
        self.media_player.durationChanged.connect(self.duration_changed)
        self.media_player.positionChanged.connect(self.update_position)
        self.media_player.error.connect(self.handle_error)

    # Create all UI elements (buttons, text boxes, etc.)
    def create_ui_components(self):
        # Video player components
        self.load_button = QPushButton("Load Video")
        self.load_button.setMinimumHeight(40)

        # Create a custom frame for highlights with ID for CSS styling
        self.highlights_frame = QFrame()
        self.highlights_frame.setObjectName("highlights_frame")  # For CSS targeting
        
        # Play/pause button with icon
        self.play_button = QPushButton()
        self.play_icon = self.style().standardIcon(QStyle.SP_MediaPlay)
        self.pause_icon = self.style().standardIcon(QStyle.SP_MediaPause)
        self.play_button.setIcon(self.play_icon)
        self.play_button.setFixedSize(40, 40)
        self.play_button.setEnabled(False)  # Disabled until a video is loaded
        self.play_button.setStyleSheet("QPushButton { color: #ffae42; }") # Orange icon color

        # Slider for navigating through the video
        self.timeline_slider = QSlider(Qt.Horizontal)
        self.timeline_slider.setRange(0, 0)
        self.timeline_slider.setTracking(True)

        # Label to show current playback time and total duration
        self.time_label = QLabel("00:00:00 / 00:00:00")

        # Clip editing buttons
        self.start_button = QPushButton("Mark Start")
        self.start_button.setMinimumHeight(40)
        self.start_button.setEnabled(False)

        self.end_button = QPushButton("Mark End")
        self.end_button.setMinimumHeight(40)
        self.end_button.setEnabled(False)

        # Preview and Save buttons
        self.preview_button = QPushButton("Preview Clip")
        self.preview_button.setMinimumHeight(40)
        self.preview_button.setEnabled(False)

        self.save_button = QPushButton("Save Clip")
        self.save_button.setMinimumHeight(40)
        self.save_button.setEnabled(False)

        # Manual time entry for precise clip control
        self.start_label = QLabel("Manual Start:")
        self.start_entry = QLineEdit()

        self.end_label = QLabel("Manual End:")
        self.end_entry = QLineEdit()

        self.apply_manual_button = QPushButton("Apply Manual Times")
        self.apply_manual_button.setMinimumHeight(40)
        self.apply_manual_button.setEnabled(False)

        # Transcription section label
        self.transcription_label = QLabel("Transcription & AI Detection")
        
        # Transcription components
        self.transcribe_button = QPushButton("Transcribe + Detect Highlights")
        self.transcribe_button.setMinimumHeight(40)
        self.transcribe_button.setEnabled(False)

        # Save Transcript button
        self.save_transcript_button = QPushButton("Save Transcript")
        self.save_transcript_button.setMinimumHeight(40)
        self.save_transcript_button.setEnabled(False)

        # Progress bar for transcription status
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)  # Hidden until transcription starts

        # Text display for transcription results
        self.result_textbox = QTextEdit()
        self.result_textbox.setReadOnly(True)
        font = QFont()
        font.setPointSize(10)
        self.result_textbox.setFont(font)

        # Highlights section
        self.highlights_label = QLabel("AI-DETECTED HIGHLIGHTS LIST")
        font = QFont()
        font.setPointSize(11)
        font.setBold(True)  # Make highlights title bold
        self.highlights_label.setFont(font)
        
        self.highlights_textbox = QTextEdit()
        self.highlights_textbox.setReadOnly(True)
        self.highlights_textbox.setFont(font)
        self.highlights_textbox.setMinimumHeight(200)

        # Buttons for handling highlights
        self.cut_clip_button = QPushButton("CUT CLIP")
        self.cut_clip_button.setMinimumHeight(40)
        self.cut_clip_button.setEnabled(False)

        self.save_clip_button = QPushButton("SAVE CLIP")
        self.save_clip_button.setMinimumHeight(40)
        self.save_clip_button.setEnabled(False)

        self.reject_button = QPushButton("REJECT")
        self.reject_button.setMinimumHeight(40)
        self.reject_button.setEnabled(False)
        
        # Add next/previous highlight navigation buttons
        self.prev_highlight_button = QPushButton("← Previous")
        self.prev_highlight_button.setMinimumHeight(40)
        self.prev_highlight_button.setEnabled(False)
        
        self.next_highlight_button = QPushButton("Next →")
        self.next_highlight_button.setMinimumHeight(40)
        self.next_highlight_button.setEnabled(False)

        # Volume control slider
        self.volume_label = QLabel("Volume:")
        self.volume_slider = QSlider(Qt.Horizontal)
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setValue(50)  # Start at 50% volume
        self.volume_slider.setFixedWidth(100)
        self.volume_slider.setTracking(True)

        # Status label to show information messages
        self.status_label = QLabel("Ready to load video :)")
   
    def create_layouts(self):
        # Main layout
        main_layout = QHBoxLayout()  # Horizontal layout
        main_layout.setSpacing(10)  # Add spacing between main sections
        
        # LEFT SECTION - Video player takes entire left side
        left_section = QVBoxLayout()
        left_section.setSpacing(8)  # Reduced spacing
        
        # Add video player to the frame with no padding
        video_frame_layout = QVBoxLayout()
        video_frame_layout.setContentsMargins(0, 0, 0, 0)
        video_frame_layout.addWidget(self.video_widget)
        self.video_frame.setLayout(video_frame_layout)
        
        # Add the frame to the left section
        left_section.addWidget(self.video_frame, 1)  # Video takes all available space
        
        # Timeline slider just below video
        timeline_layout = QHBoxLayout()
        timeline_layout.addWidget(self.timeline_slider)
        left_section.addLayout(timeline_layout)
        
        # Video playback controls at the bottom
        playback_controls = QHBoxLayout()
        playback_controls.setSpacing(10)  # Add spacing between controls
        playback_controls.addWidget(self.play_button)
        playback_controls.addWidget(self.time_label)
        
        # Volume controls
        volume_layout = QHBoxLayout()
        volume_layout.addWidget(self.volume_label)
        volume_layout.addWidget(self.volume_slider)
        volume_layout.setSpacing(5)  # Reduce spacing between label and slider
        playback_controls.addLayout(volume_layout)
        
        left_section.addLayout(playback_controls)
        
        # Status bar at the bottom of left section
        status_layout = QHBoxLayout()
        status_layout.addWidget(self.status_label)
        left_section.addLayout(status_layout)
        
        # RIGHT SECTION - Controls and highlights
        right_section = QVBoxLayout()
        right_section.setSpacing(12)  # More spacing between control groups
        
        # Load video button
        right_section.addWidget(self.load_button)
        
        # Clip editing controls - two buttons per row
        mark_buttons = QHBoxLayout()
        mark_buttons.addWidget(self.start_button)
        mark_buttons.addWidget(self.end_button)
        right_section.addLayout(mark_buttons)
        
        preview_save_buttons = QHBoxLayout()
        preview_save_buttons.addWidget(self.preview_button)
        preview_save_buttons.addWidget(self.save_button)
        right_section.addLayout(preview_save_buttons)
        
        # Manual time entry section - Put inputs next to labels
        start_time_layout = QHBoxLayout()
        start_time_layout.addWidget(self.start_label)
        start_time_layout.addWidget(self.start_entry)
        
        end_time_layout = QHBoxLayout()
        end_time_layout.addWidget(self.end_label)
        end_time_layout.addWidget(self.end_entry)
        
        right_section.addLayout(start_time_layout)
        right_section.addLayout(end_time_layout)
        right_section.addWidget(self.apply_manual_button)
        
        # Transcription section
        right_section.addWidget(self.transcription_label)
        
        # Transcription controls
        transcription_buttons = QHBoxLayout()
        transcription_buttons.addWidget(self.transcribe_button)
        transcription_buttons.addWidget(self.save_transcript_button)
        right_section.addLayout(transcription_buttons)
        right_section.addWidget(self.progress_bar)
        
        # Highlight navigation buttons
        navigation_buttons = QHBoxLayout()
        navigation_buttons.addWidget(self.prev_highlight_button)
        navigation_buttons.addWidget(self.next_highlight_button)
        right_section.addLayout(navigation_buttons)
        
        # Highlight action buttons
        highlight_buttons = QHBoxLayout()
        highlight_buttons.addWidget(self.cut_clip_button)
        highlight_buttons.addWidget(self.save_clip_button)
        highlight_buttons.addWidget(self.reject_button)
        right_section.addLayout(highlight_buttons)
        
        # MOVED: AI-Detected Highlights section to right panel
        right_section.addWidget(self.highlights_label)
        right_section.addWidget(self.highlights_textbox, 1)  # Give this stretch factor to fill space
        
        # Set fixed width for right section - wider by about an inch (100 pixels)
        right_widget = QWidget()
        right_widget.setLayout(right_section)
        right_widget.setFixedWidth(450)  # Increased from 350
        
        # Add left and right sections to main layout
        main_layout.addLayout(left_section, 1)  # Video side gets all remaining space
        main_layout.addWidget(right_widget, 0)  # Right side has fixed width
        
        # Set the main layout
        self.setLayout(main_layout)   
    def setup_connections(self):
        # Video loading and control
        self.load_button.clicked.connect(self.load_media)
        self.play_button.clicked.connect(self.toggle_play)
        self.timeline_slider.sliderMoved.connect(self.seek_position)
        self.volume_slider.valueChanged.connect(self.change_volume)

        # Clip editing
        self.start_button.clicked.connect(self.mark_start)
        self.end_button.clicked.connect(self.mark_end)
        self.preview_button.clicked.connect(self.preview_clip)
        self.save_button.clicked.connect(self.save_clip)
        self.apply_manual_button.clicked.connect(self.set_manual_times)

        # Transcription
        self.transcribe_button.clicked.connect(self.transcribe_video)
        self.save_transcript_button.clicked.connect(self.save_transcript)

        # Highlight management
        self.cut_clip_button.clicked.connect(self.handle_highlight_cut)
        self.save_clip_button.clicked.connect(self.handle_highlight_save)
        self.reject_button.clicked.connect(self.handle_highlight_reject)
        
        # Highlight navigation
        self.prev_highlight_button.clicked.connect(self.go_to_previous_highlight)
        self.next_highlight_button.clicked.connect(self.go_to_next_highlight)

        # Special event for double-clicking on highlights
        self.highlights_textbox.mouseDoubleClickEvent = self.highlight_double_clicked

    # Apply a dark color theme to the application
    def apply_dark_theme(self):
        flux_stylesheet = """
        QWidget {
            background-color: #2b1d0e; /* dark brown */
            color: #ffae42; /* soft orange text */
            border-radius: 3px; /* Very slight rounded corners everywhere */
        }
        QPushButton {
            background-color: #3c2a17; /* slightly lighter brown */
            color: #ffae42;
            border: 1px solid #5c3b1c;
            padding: 5px;
            border-radius: 4px;
            min-height: 30px; /* Standardize button heights */
            min-width: 80px; /* Set minimum width for buttons */
        }
        QPushButton:hover {
            background-color: #5c3b1c; /* hover lighter brown */
        }
        QPushButton:disabled {
            background-color: #2b1d0e;
            color: #7f5a2e;
        }
        QLineEdit, QTextEdit {
            background-color: #3c2a17;
            color: #ffae42;
            border: 1px solid #5c3b1c;
            border-radius: 4px;
            padding: 3px;
        }
        QLineEdit {
            max-height: 25px; /* Reduce height of timestamp boxes */
        }
        QProgressBar {
            background-color: #3c2a17;
            color: #ffae42;
            border: 1px solid #5c3b1c;
            border-radius: 4px;
            text-align: center;
        }
        QProgressBar::chunk {
            background-color: #ffae42;
            width: 10px;
        }
        QSlider::groove:horizontal {
            border: 1px solid #5c3b1c;
            height: 8px;
            background: #3c2a17;
            border-radius: 3px;
        }
        QSlider::handle:horizontal {
            background: #ffae42;
            border: 1px solid #5c3b1c;
            width: 18px;
            margin: -5px 0;
            border-radius: 9px;
        }
        QLabel {
            color: #ffae42;
            padding: 2px;
        }
        QFrame {
            border: 1px solid #5c3b1c;
            border-radius: 5px;
        }
        QFrame#highlights_frame {
            background-color: #2f1f0f; /* Slightly different background to make highlights stand out */
            border: 1px solid #5c3b1c;
            border-radius: 5px;
            margin-top: 10px;
            padding: 5px;
        }
        """
        self.setStyleSheet(flux_stylesheet)  # Apply the style to the application

    # Function to load a video file
    def load_media(self):
        file_dialog = QFileDialog(self)
        file_dialog.setFileMode(QFileDialog.ExistingFile)
        file_dialog.setNameFilter("Media Files (*.mp4 *.mov *.avi *.mkv *.mp3 *.wav)")
        if file_dialog.exec_():
            filepath = file_dialog.selectedFiles()[0]
            if not os.path.exists(filepath):
                QMessageBox.critical(self, "Error", "Selected file does not exist.")
                return

            self.video_file_path = filepath
            self.audio_path = filepath

            # Determine if it's audio or video based on file extension
            self.is_audio_only = self.is_audio_file(filepath)

            try:
                # Always set the media content
                self.media_player.setMedia(QMediaContent(QUrl.fromLocalFile(filepath)))
                
                if self.is_audio_only:
                    # Create and add visualizer in place of video display
                    if LIBROSA_AVAILABLE:
                        self.visualizer_container, self.waveform = self.create_audio_visualizer(
                            filepath, 
                            self.video_widget.width(), 
                            self.video_widget.height()
                        )
                        
                        # Replace video widget with audio visualizer
                        layout_index = self.video_frame.layout().indexOf(self.video_widget)
                        self.video_frame.layout().replaceWidget(self.video_widget, self.visualizer_container)
                        self.video_widget.hide()
                        self.visualizer_container.show()
                    
                    self.video_clip = None
                else:
                    # It's a video file - make sure we show the video widget
                    if hasattr(self, 'visualizer_container') and self.visualizer_container is not None:
                        layout_index = self.video_frame.layout().indexOf(self.visualizer_container)
                        self.video_frame.layout().replaceWidget(self.visualizer_container, self.video_widget)
                        self.visualizer_container.hide()
                    self.video_widget.show()
                    
                    # Load video file with MoviePy
                    self.video_clip = VideoFileClip(filepath)
                
                # Make sure video elements have proper size
                self.video_widget.setMinimumHeight(360)
                
                self.status_label.setText(f"Loaded: {os.path.basename(filepath)}")
                self.enable_controls(True)
                self.clip_start_time = None
                self.clip_end_time = None
                self.start_entry.clear()
                self.end_entry.clear()
                self.highlights_textbox.clear()
                self.highlights.clear()
                self.current_highlight_index = -1
                self.transcribe_button.setEnabled(True)
                self.prev_highlight_button.setEnabled(False)
                self.next_highlight_button.setEnabled(False)
                
            except Exception as e:
                QMessageBox.critical(self, "Load Error", f"Failed to load media: {str(e)}")
                return

    # Toggle between play and pause
    def toggle_play(self):
        if self.media_player.state() == QMediaPlayer.PlayingState:
            self.media_player.pause()  # Pause if playing
        else:
            self.media_player.play()   # Play if paused

    # Update waveform and position when media position changes
    def update_position(self, position):
        # Update waveform position if it exists
        if self.is_audio_only and self.waveform is not None:
            self.waveform.set_position(position / 1000.0)  # Convert ms to seconds

    # Add mouse tracking for highlight timestamp hovering
    def highlight_mouse_move(self, event):
        cursor = self.highlights_textbox.cursorForPosition(event.pos())
        cursor.select(QTextCursor.LineUnderCursor)
        line = cursor.selectedText()
        
        # Change cursor if over a timestamp line
        if "Highlight #" in line:
            self.highlights_textbox.viewport().setCursor(Qt.PointingHandCursor)
        else:
            self.highlights_textbox.viewport().setCursor(Qt.IBeamCursor)
            
        # Pass event to parent for normal handling
        QTextEdit.mouseMoveEvent(self.highlights_textbox, event)

    def media_state_changed(self, state):
        if state == QMediaPlayer.PlayingState:
            self.play_button.setIcon(self.pause_icon)  # Show pause icon
            self.is_playing = True
            self.update_timer.start()  # Start the timer that updates time display
        else:
            self.play_button.setIcon(self.play_icon)  # Show play icon
            self.is_playing = False
            self.update_timer.stop()   # Stop the update timer

    # Handle when the video duration is determined
    def duration_changed(self, duration):
        duration_sec = duration / 1000  # Convert milliseconds to seconds
        self.timeline_slider.setRange(0, duration)  # Set the slider range
        self.time_label.setText(f"00:00:00 / {self.format_time(duration_sec)}")

    # Update the playback position display while video is playing
    def update_playback_position(self):
        if self.is_playing:
            position = self.media_player.position()  # Get current position in ms
            # Update the slider without triggering signals
            self.timeline_slider.blockSignals(True)
            self.timeline_slider.setValue(position)
            self.timeline_slider.blockSignals(False)
            current_sec = position / 1000  # Convert to seconds
            duration_sec = self.media_player.duration() / 1000
            # Update the time display
            self.time_label.setText(f"{self.format_time(current_sec)} / {self.format_time(duration_sec)}")
            self.current_time = current_sec

    # Jump to a position in the video when slider is moved
    def seek_position(self, position):
        self.media_player.setPosition(position)
        self.current_time = position / 1000
        duration_sec = self.media_player.duration() / 1000
        self.time_label.setText(f"{self.format_time(self.current_time)} / {self.format_time(duration_sec)}")

    # Adjust volume when slider is moved
    def change_volume(self, value):
        self.media_player.setVolume(value)

    # ===== CLIP EDITOR FUNCTIONS =====
    # Mark the current position as the start of a clip
    def mark_start(self):
        self.clip_start_time = self.current_time
        self.start_entry.setText(self.format_time(self.current_time))
        self.status_label.setText(f"Start marked at {self.format_time(self.current_time)}")
        self.update_clip_controls()

    # Mark the current position as the end of a clip
    def mark_end(self):
        self.clip_end_time = self.current_time
        self.end_entry.setText(self.format_time(self.current_time))
        self.status_label.setText(f"End marked at {self.format_time(self.current_time)}")
        self.update_clip_controls()

    # Set clip times manually from text input
    def set_manual_times(self):
        try:
            # Convert the text entries to seconds
            start_time = self.parse_time_string(self.start_entry.text())
            end_time = self.parse_time_string(self.end_entry.text())
            
            # Check if times are valid for video
            if not self.is_audio_only:
                if start_time >= 0 and end_time > start_time and end_time <= self.video_clip.duration:
                    self.clip_start_time = start_time
                    self.clip_end_time = end_time
                    self.status_label.setText(f"Manual times set: {self.format_time(start_time)} to {self.format_time(end_time)}")
                    self.update_clip_controls()
                else:
                    raise ValueError("Invalid time range")
            # For audio files
            else:
                if start_time >= 0 and end_time > start_time:
                    self.clip_start_time = start_time
                    self.clip_end_time = end_time
                    self.status_label.setText(f"Manual times set: {self.format_time(start_time)} to {self.format_time(end_time)}")
                    self.update_clip_controls()
                else:
                    raise ValueError("Invalid time range")
        except ValueError:
            QMessageBox.warning(self, "Invalid Time", "Please enter valid times in HH:MM:SS format.")

    # Add a preview clip function
    def preview_clip(self):
        if self.validate_clip_times():
            # Just seek to the start time and play
            self.media_player.setPosition(int(self.clip_start_time * 1000))
            self.media_player.play()
            
            # Optional: Set a timer to stop at the end time
            end_time_ms = int(self.clip_end_time * 1000)
            start_time_ms = int(self.clip_start_time * 1000)
            duration_ms = end_time_ms - start_time_ms
            
            # Create a one-shot timer to stop playback
            QTimer.singleShot(duration_ms, self.media_player.pause)

    # Save the selected clip as a new video file
    def save_clip(self):
        if self.validate_clip_times():
            try:
                if self.is_audio_only:
                    QMessageBox.warning(self, "Audio Only", "Saving clips is only supported for videos right now.")
                    return

                original_filename = os.path.basename(self.video_file_path)
                name_without_ext = os.path.splitext(original_filename)[0]
                default_output = f"{name_without_ext}_clip_{self.format_time(self.clip_start_time)}-{self.format_time(self.clip_end_time)}.mp4"

                output_path, _ = QFileDialog.getSaveFileName(self, "Save Clip", os.path.join(QDir.homePath(), default_output), "Video Files (*.mp4)")
                if output_path:
                    self.status_label.setText("Saving clip... Please wait")
                    QApplication.processEvents()

                    subclip = self.video_clip.subclip(self.clip_start_time, self.clip_end_time)
                    subclip.write_videofile(output_path, codec='libx264', audio_codec='aac', preset='medium', threads=4)
                    self.status_label.setText(f"Clip saved: {os.path.basename(output_path)}")
            except Exception as e:
                QMessageBox.critical(self, "Save Error", f"Failed to save clip: {str(e)}")

    # Check if clip times are valid
    def validate_clip_times(self):
        if self.clip_start_time is None or self.clip_end_time is None:
            QMessageBox.warning(self, "Missing Time Markers", "Please mark both start and end times.")
            return False
        if self.clip_end_time <= self.clip_start_time:
            QMessageBox.warning(self, "Invalid Time Range", "End time must be after start time.")
            return False
        if not self.is_audio_only and (self.clip_start_time < 0 or self.clip_end_time > self.video_clip.duration):
            QMessageBox.warning(self, "Out of Range", "Clip times must be within video duration.")
            return False
        return True
 
 
    # ===== TRANSCRIPTION + AI DETECTION =====
    # Start the transcription and highlight detection process
    def transcribe_video(self):
        if not self.audio_path:
            QMessageBox.warning(self, "No Media", "Please load a video or audio file before transcribing.")
            return

        # Update UI to show processing is starting
        self.status_label.setText("Transcribing and analyzing... please wait.")
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.result_textbox.clear()
        self.highlights_textbox.clear()
        self.highlights.clear()
        self.current_highlight_index = -1
        QApplication.processEvents()  # Update the UI immediately

        # Reset text variables
        self.full_text = ""
        self.pending_segments.clear()

        # Create and start the worker thread for transcription
        self.worker = TranscriptionWorker(self.model, self.audio_path)
        self.worker.progress.connect(self.update_progress)
        self.worker.live_update.connect(self.animate_typing)
        self.worker.finished.connect(self.handle_transcription_finished)
        self.worker.start()

    # Update the progress bar when transcription advances
    def update_progress(self, value):
        self.progress_bar.setValue(value)

    # Create a typing animation for new text segments
    def animate_typing(self, new_text):
        if self.typing_timer.isActive():
            # If already typing, add this to queue
            self.pending_segments.append(new_text)
        else:
            # Start typing this text
            self.current_typing_text = new_text
            self.current_char_index = 0
            self.typing_timer.start(20)  # Type a character every 20ms

    # Type one character at a time for a more natural appearance
    def type_next_character(self):
        if self.current_char_index < len(self.current_typing_text):
            # Add the next character to the full text
            self.full_text += self.current_typing_text[self.current_char_index]
            self.result_textbox.setPlainText(self.full_text)
            self.result_textbox.moveCursor(QTextCursor.End)  # Scroll to end
            self.current_char_index += 1
        else:
            # Finished typing this segment
            self.typing_timer.stop()
            if self.pending_segments:
                # Start typing the next segment if any
                next_text = self.pending_segments.pop(0)
                self.current_typing_text = next_text
                self.current_char_index = 0
                self.typing_timer.start(20)

    # Handle when the transcription and highlight detection is finished
    def handle_transcription_finished(self, detected_highlights):
        self.highlights = detected_highlights
        
        # Group highlights by emotion type for summary
        emotion_counts = {}
        for _, _, _, emotion in detected_highlights:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        if emotion_counts:
            emotion_summary = ", ".join([f"{count} {emotion}" for emotion, count in emotion_counts.items()])
            summary = f"Transcription complete! {len(detected_highlights)} highlights detected: {emotion_summary}"
        else:
            summary = f"Transcription complete! {len(detected_highlights)} highlights detected."
            
        self.status_label.setText(summary)
        self.progress_bar.setVisible(False)
        self.save_transcript_button.setEnabled(True)
        
        # Display highlights in the highlights text box
        if detected_highlights:
            self.display_highlights()
            self.cut_clip_button.setEnabled(True)
            self.save_clip_button.setEnabled(True)
            self.reject_button.setEnabled(True)
            self.prev_highlight_button.setEnabled(True)
            self.next_highlight_button.setEnabled(True)
        else:
            self.highlights_textbox.setPlainText("No highlights detected in this video.")
            
        # Complete typing any remaining text
        if self.pending_segments:
            for segment in self.pending_segments:
                self.full_text += segment
            self.result_textbox.setPlainText(self.full_text)
            self.result_textbox.moveCursor(QTextCursor.End)
            self.pending_segments.clear()

    # Display highlights in the highlights text box
    def display_highlights(self):
        self.highlights_textbox.clear()
        
        # Format for improved timestamp visibility
        for i, (start, end, text, emotion) in enumerate(self.highlights):
            # Create a timestamp that stands out
            highlight_num = i + 1
            timestamp_text = f"Highlight #{highlight_num}: {self.format_time(start)} to {self.format_time(end)}"
            
            # Format the text with HTML to style the timestamp differently
            # Use underline and different color for timestamp
            formatted_text = f'<span style="color:#ff9933; text-decoration:underline;">{timestamp_text}</span><br>'
            formatted_text += f"Text: {text}<br>"
            formatted_text += "-" * 50 + "<br><br>"
            
            self.highlights_textbox.append(formatted_text)

    # Navigate to the next highlight in the list
    def go_to_next_highlight(self):
        if not self.highlights:
            return
            
        # Move to the next highlight
        if self.current_highlight_index < len(self.highlights) - 1:
            self.current_highlight_index += 1
        else:
            # Wrap around to the first highlight
            self.current_highlight_index = 0
            
        # Jump to this highlight
        self.jump_to_current_highlight()
        
    # Navigate to the previous highlight in the list
    def go_to_previous_highlight(self):
        if not self.highlights:
            return
            
        # Move to the previous highlight
        if self.current_highlight_index > 0:
            self.current_highlight_index -= 1
        else:
            # Wrap around to the last highlight
            self.current_highlight_index = len(self.highlights) - 1
            
        # Jump to this highlight
        self.jump_to_current_highlight()
        
    # Jump to the currently selected highlight
    def jump_to_current_highlight(self):
        if 0 <= self.current_highlight_index < len(self.highlights):
            start_time, end_time, text, emotion = self.highlights[self.current_highlight_index]
            
            # Jump to the start time in the video
            self.media_player.setPosition(int(start_time * 1000))
            self.current_time = start_time
            
            # Set this as the current clip start/end times
            self.clip_start_time = start_time
            self.clip_end_time = end_time
            self.start_entry.setText(self.format_time(start_time))
            self.end_entry.setText(self.format_time(end_time))
            self.update_clip_controls()
            
            # Highlight the text in the textbox
            self.highlight_in_textbox(self.current_highlight_index + 1)
            
            self.status_label.setText(f"Viewing {emotion} highlight #{self.current_highlight_index+1}/{len(self.highlights)}")

    # Highlight the specified highlight number in the text box
    def highlight_in_textbox(self, highlight_num):
        # Find the highlight in the text box
        highlight_text = f"Highlight #{highlight_num}:"
        
        # Get the current text
        full_text = self.highlights_textbox.toPlainText()
        
        # Find the position of the highlight
        cursor = self.highlights_textbox.textCursor()
        cursor.setPosition(0)
        self.highlights_textbox.setTextCursor(cursor)
        
        # Use the find method to locate and select the highlight
        if self.highlights_textbox.find(highlight_text):
            # The text is now selected - make sure it's visible
            self.highlights_textbox.ensureCursorVisible()

    # Save the full transcript to a text file
    def save_transcript(self):
        if not self.full_text:
            QMessageBox.warning(self, "No Transcript", "There is no transcript to save.")
            return
            
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Transcript", 
            os.path.join(QDir.homePath(), f"{os.path.splitext(os.path.basename(self.video_file_path))[0]}_transcript.txt"),
            "Text Files (*.txt)"
        )
        
        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(self.full_text)
                self.status_label.setText(f"Transcript saved to {os.path.basename(filename)}")
            except Exception as e:
                QMessageBox.critical(self, "Save Error", f"Failed to save transcript: {str(e)}")

    # Handle double-click on a highlight entry
    def highlight_double_clicked(self, event):
        cursor = self.highlights_textbox.cursorForPosition(event.pos())
        cursor.select(QTextCursor.LineUnderCursor)
        selected_line = cursor.selectedText()
        
        # Check if the selected line contains a highlight time
        if "Highlight #" in selected_line:
            try:
                # Extract highlight number
                highlight_num = int(selected_line.split("Highlight #")[1].split(":")[0]) - 1
                if 0 <= highlight_num < len(self.highlights):
                    # Set as the current highlight index and jump to it
                    self.current_highlight_index = highlight_num
                    self.jump_to_current_highlight()
            except Exception as e:
                print(f"Error processing highlight click: {e}")

    # Handle cutting the current highlight directly
    def handle_highlight_cut(self):
        if not self.highlights or self.current_highlight_index < 0:
            return
            
        # Get currently selected highlight
        if 0 <= self.current_highlight_index < len(self.highlights):
            start_time, end_time, title, emotion = self.highlights[self.current_highlight_index]
            self.clip_start_time = start_time
            self.clip_end_time = end_time
            self.save_clip()

    # Handle saving the current highlight
    def handle_highlight_save(self):
        if not self.highlights or self.current_highlight_index < 0:
            return
            
        # Similar to cut but just sets the times without saving
        if 0 <= self.current_highlight_index < len(self.highlights):
            start_time, end_time, title, emotion = self.highlights[self.current_highlight_index]
            self.clip_start_time = start_time
            self.clip_end_time = end_time
            self.start_entry.setText(self.format_time(start_time))
            self.end_entry.setText(self.format_time(end_time))
            self.update_clip_controls()
            self.status_label.setText(f"Set times to {emotion} highlight #{self.current_highlight_index+1}")

    # Handle rejecting/removing a highlight
    def handle_highlight_reject(self):
        if not self.highlights or self.current_highlight_index < 0:
            return
            
        # Remove the current highlight
        if 0 <= self.current_highlight_index < len(self.highlights):
            # Remove the highlight
            self.highlights.pop(self.current_highlight_index)
            
            # Update display
            self.display_highlights()
            
            # Handle case where all highlights are removed
            if not self.highlights:
                self.cut_clip_button.setEnabled(False)
                self.save_clip_button.setEnabled(False)
                self.reject_button.setEnabled(False)
                self.prev_highlight_button.setEnabled(False)
                self.next_highlight_button.setEnabled(False)
                self.highlights_textbox.setPlainText("All highlights have been reviewed.")
                self.current_highlight_index = -1
            else:
                # Adjust current index if needed
                if self.current_highlight_index >= len(self.highlights):
                    self.current_highlight_index = len(self.highlights) - 1
                
                # Jump to new current highlight
                self.jump_to_current_highlight()

    # Update which clip control buttons are enabled based on current state
    def update_clip_controls(self):
        can_save = (self.clip_start_time is not None and 
                    self.clip_end_time is not None and 
                    self.clip_end_time > self.clip_start_time)
        self.save_button.setEnabled(can_save)
        self.preview_button.setEnabled(can_save)

    # Format time in seconds to HH:MM:SS format
    def format_time(self, seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    # Parse a time string in HH:MM:SS format to seconds
    def parse_time_string(self, time_str):
        parts = time_str.split(':')
        if len(parts) == 3:
            hours, minutes, seconds = map(int, parts)
            return hours * 3600 + minutes * 60 + seconds
        elif len(parts) == 2:
            minutes, seconds = map(int, parts)
            return minutes * 60 + seconds
        else:
            try:
                return int(time_str)
            except ValueError:
                raise ValueError("Invalid time format")

    # Enable or disable controls based on whether media is loaded
    def enable_controls(self, enabled):
        self.play_button.setEnabled(enabled)
        self.start_button.setEnabled(enabled)
        self.end_button.setEnabled(enabled)
        self.preview_button.setEnabled(enabled)
        self.apply_manual_button.setEnabled(enabled)
        self.transcribe_button.setEnabled(enabled)
        
    # Handle errors in the media player
    def handle_error(self):
        error_message = self.media_player.errorString()
        self.status_label.setText(f"Error: {error_message}")
        QMessageBox.critical(self, "Media Error", f"An error occurred: {error_message}")

# Main entry point for the application
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoTranscriberEditor()
    window.show()
    sys.exit(app.exec_())