import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import numpy as np

# Use MediaPipe via WASM in the browser:
from streamlit_webrtc import webrtc_ctx
import av

# Initialize session state keys
if 'page' not in st.session_state:
    st.session_state.page = "Home"
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False
if 'gesture_data' not in st.session_state:
    st.session_state.gesture_data, st.session_state.mappings = {}, {}
if 'model' not in st.session_state:
    st.session_state.model = None
    st.session_state.model_type = None
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False    

# Initialize MediaPipe Hands with enhanced configuration
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8
)


# Global variables for action cooldown and state
last_action_time = 0
ACTION_COOLDOWN = 1.0  # seconds
current_volume = 50  # Track system volume state

# Model options
MODEL_OPTIONS = {
    "SVM": "Support Vector Machine",
    "CNN": "1D Convolutional Neural Network",
    "LSTM": "Long Short-Term Memory"
}

# Enhanced CSS styling
def load_css():
    st.markdown(f"""
    <style>
    /* Main container */
    .main {{
        background-color: {'#0e1117' if st.session_state.dark_mode else '#ffffff'};
        color: {'#ffffff' if st.session_state.dark_mode else '#0e1117'};
    }}
    
    /* Sidebar */
    .sidebar .sidebar-content {{
        background-color: {'#1a1a1a' if st.session_state.dark_mode else '#f0f2f6'} !important;
    }}
    
    /* Buttons */
    .stButton>button {{
        border: 2px solid {'#4CAF50' if st.session_state.dark_mode else '#2e7d32'};
        background-color: {'#2e7d32' if st.session_state.dark_mode else '#4CAF50'};
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }}
    .stButton>button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }}
    
    /* Cards */
    .card {{
        background-color: {'#1e1e1e' if st.session_state.dark_mode else '#f8f9fa'};
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }}
    .card:hover {{
        transform: translateY(-5px);
        box-shadow: 0 8px 15px rgba(0,0,0,0.2);
    }}
    
    /* Progress bars */
    .stProgress > div > div > div {{
        background-color: {'#4CAF50' if st.session_state.dark_mode else '#2e7d32'};
    }}
    
    /* Custom animated header */
    @keyframes gradient {{
        0% {{ background-position: 0% 50%; }}
        50% {{ background-position: 100% 50%; }}
        100% {{ background-position: 0% 50%; }}
    }}
    .animated-header {{
        background: linear-gradient(-45deg, #4CAF50, #2e7d32, #4CAF50, #2e7d32);
        background-size: 400% 400%;
        animation: gradient 15s ease infinite;
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }}
    </style>
    """, unsafe_allow_html=True)

def numpy_to_serializable(data):
    """Convert numpy arrays to JSON-serializable formats"""
    if isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, dict):
        return {k: numpy_to_serializable(v) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return [numpy_to_serializable(item) for item in data]
    return data

def extract_landmarks(hand_landmarks):
    if not hand_landmarks:
        return None
        
    base = hand_landmarks.landmark[0]  # Wrist joint
    
    # Existing relative coordinates
    landmarks = []
    for landmark in hand_landmarks.landmark:
        landmarks.extend([landmark.x - base.x, 
                         landmark.y - base.y, 
                         landmark.z - base.z])
    
    # NEW: Advanced features
    thumb_tip = hand_landmarks.landmark[4]
    index_tip = hand_landmarks.landmark[8]
    
    # Distance between thumb and index
    pinch_distance = np.sqrt(
        (thumb_tip.x - index_tip.x)**2 + 
        (thumb_tip.y - index_tip.y)**2 + 
        (thumb_tip.z - index_tip.z)**2
    )
    
    # Palm width (MCP joints)
    palm_width = np.sqrt(
        (hand_landmarks.landmark[5].x - hand_landmarks.landmark[17].x)**2 +
        (hand_landmarks.landmark[5].y - hand_landmarks.landmark[17].y)**2
    )
    
    # Append new features
    landmarks.extend([pinch_distance, palm_width])
    
    return np.array(landmarks)

def record_gesture_samples(gesture_name, num_samples=50):
    """Record gesture samples with visual feedback"""
    cap = cv2.VideoCapture(0)
    samples = []
    sample_count = 0
    progress_bar = st.progress(0)
    status_text = st.empty()
    frame_placeholder = st.empty()
    
    while sample_count < num_samples and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue
            
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = extract_landmarks(hand_landmarks)
                if landmarks is not None:
                    samples.append(landmarks)
                    sample_count += 1
                    progress_bar.progress(sample_count / num_samples)
                    status_text.text(f"Recording: {sample_count}/{num_samples}")
                    
                    # Enhanced visualization
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
                    
                    # Countdown overlay
                    cv2.putText(frame, f"Hold gesture... {num_samples-sample_count} left", 
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        frame_placeholder.image(frame, channels="BGR", use_column_width=True)
        time.sleep(0.1)
    
    cap.release()
    progress_bar.empty()
    status_text.empty()
    return samples

def prepare_training_data(gesture_data):
    """Prepare training data for neural networks (CNN/LSTM)"""
    X, y = [], []
    gesture_names = list(gesture_data.keys())
    
    # Create balanced dataset (use same number of samples per gesture)
    min_samples = min(len(samples) for samples in gesture_data.values())
    
    for gesture_name, samples in gesture_data.items():
        selected_samples = samples[:min_samples]
        X.extend(selected_samples)
        y.extend([gesture_names.index(gesture_name)] * len(selected_samples))
    
    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    # Data augmentation for neural networks
    augmented_X = []
    augmented_y = []
    
    for sample, label in zip(X, y):
        # Original sample
        augmented_X.append(sample)
        augmented_y.append(label)
        
        # Add noisy version (15% random noise)
        noise = np.random.normal(0, 0.15, sample.shape) * (np.max(sample) - np.min(sample))
        augmented_X.append(sample + noise)
        augmented_y.append(label)
        
        # Add scaled version (90-110% scale)
        scale = np.random.uniform(0.9, 1.1)
        augmented_X.append(sample * scale)
        augmented_y.append(label)
    
    X = np.array(augmented_X)
    y = np.array(augmented_y)
    
    # Reshape for neural networks (samples, timesteps, features)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    
    # One-hot encode labels
    y = to_categorical(y, len(gesture_names))
    
    return X, y
def should_update_model(gesture_data, existing_model_type):
    """Check if we can update the existing model with new data"""
    if not st.session_state.model or not existing_model_type:
        return False
    
    # Check if the new data has the same gestures as before
    old_gestures = set(st.session_state.gesture_data.keys())
    new_gestures = set(gesture_data.keys())
    
    # If new gestures were added, we need to retrain from scratch
    if not new_gestures.issubset(old_gestures):
        return False
    
    return True

def create_model(model_type, input_shape, num_classes):
    """Create different model architectures"""
    if model_type == "SVM":
        return make_pipeline(
            StandardScaler(),
            SVC(kernel='rbf', C=10, gamma='scale', probability=True)
        )
    elif model_type == "CNN":
        model = Sequential([
            Conv1D(64, 3, activation='relu', input_shape=input_shape),
            MaxPooling1D(2),
            Conv1D(128, 3, activation='relu'),
            MaxPooling1D(2),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(num_classes, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model
    elif model_type == "LSTM":
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=input_shape),
            LSTM(128),
            Dense(64, activation='relu'),
            Dense(num_classes, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model
    

def plot_training_history(history, model_type):
    """Plot training metrics"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Accuracy plot
    ax1.plot(history.history['accuracy'], label='Train Accuracy')
    if 'val_accuracy' in history.history:
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title(f'{model_type} Accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend()
    
    # Loss plot
    ax2.plot(history.history['loss'], label='Train Loss')
    if 'val_loss' in history.history:
        ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title(f'{model_type} Loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend()
    
    st.pyplot(fig)

def plot_confusion_matrix(y_true, y_pred, classes):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    st.pyplot(plt)

def train_gesture_model(gesture_data, model_type, epochs=10, incremental=True):
    """Train selected model type with visualization"""
    if not gesture_data or len(gesture_data) < 2:
        st.error("Need at least 2 different gestures to train")
        return None
        
    # Prepare data
    X, y = [], []
    gesture_names = list(gesture_data.keys())
    min_samples = min(len(samples) for samples in gesture_data.values())
    
    for gesture_name, samples in gesture_data.items():
        selected_samples = samples[:min_samples]
        X.extend(selected_samples)
        y.extend([gesture_name] * len(selected_samples))
    
    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42)
    
    # =============================================
    # NEW: DATA AUGMENTATION FOR NEURAL NETWORKS
    # =============================================
    if model_type in ["CNN", "LSTM"]:
        augmented_samples = []
        augmented_labels = []
        
        for sample, label in zip(X_train, y_train):
            # Original sample
            augmented_samples.append(sample)
            augmented_labels.append(label)
            
            # Add noisy version (15% random noise)
            noise = np.random.normal(0, 0.15, sample.shape) * (np.max(sample) - np.min(sample))
            augmented_samples.append(sample + noise)
            augmented_labels.append(label)
            
            # Add scaled version (90-110% scale)
            scale = np.random.uniform(0.9, 1.1)
            augmented_samples.append(sample * scale)
            augmented_labels.append(label)
        
        X_train = np.array(augmented_samples)
        y_train = np.array(augmented_labels)
    # =============================================
    
    # Check if we can update existing model
    can_update = (st.session_state.model is not None and 
                 st.session_state.model_type == model_type and
                 set(gesture_names).issubset(set(getattr(st.session_state, 'gesture_names', []))))
    
    # Model-specific processing
    if model_type == "SVM":
        if can_update:
            st.info("New data detected - retraining SVM from scratch")
        model = create_model(model_type, None, None)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        st.success(f"SVM trained successfully! Accuracy: {accuracy*100:.1f}%")
        st.subheader("Classification Report")
        st.text(classification_report(y_test, y_pred, target_names=gesture_names))
        
        st.subheader("Confusion Matrix")
        plot_confusion_matrix(y_test, y_pred, gesture_names)
        
    else:  # For neural networks
        # Reshape for neural networks
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
        input_shape = (X_train.shape[1], 1)
        
        # One-hot encode for neural networks
        y_train_cat = to_categorical(y_train, len(gesture_names))
        y_test_cat = to_categorical(y_test, len(gesture_names))
        
        if can_update:
            st.info("Updating existing model with new data")
            model = st.session_state.model
            # Reduce learning rate for fine-tuning
            tf.keras.backend.set_value(model.optimizer.learning_rate, 0.0001)
            epochs = max(5, epochs // 2)  # Use fewer epochs for updates
        else:
            model = create_model(model_type, input_shape, len(gesture_names))
        
        st.write(f"Training {model_type} model...")
        history = model.fit(
            X_train, y_train_cat,
            validation_data=(X_test, y_test_cat),
            epochs=epochs,
            batch_size=32,
            verbose=1
        )
        
        # Evaluate
        loss, accuracy = model.evaluate(X_test, y_test_cat)
        st.success(f"{model_type} trained successfully! Accuracy: {accuracy*100:.1f}%")
        
        # Plot training history
        st.subheader("Training Metrics")
        plot_training_history(history, model_type)
        
        # Confusion matrix
        st.subheader("Confusion Matrix")
        y_pred = np.argmax(model.predict(X_test), axis=1)
        plot_confusion_matrix(y_test, y_pred, gesture_names)
    
        # Save model and update session state
        model_path = f"gesture_model_{model_type}.h5" if model_type != "SVM" else f"gesture_model_{model_type}.pkl"
        try:
            if model_type == "SVM":
                joblib.dump(model, model_path)
            else:
                # Clear TensorFlow session and remove existing file
                tf.keras.backend.clear_session()
                if os.path.exists(model_path):
                    os.remove(model_path)
                time.sleep(0.1)  # Small delay for filesystem
                model.save(model_path, save_format='h5')  # Explicit HDF5 format
            
            st.session_state.model = model
            st.session_state.model_type = model_type
            st.session_state.gesture_names = gesture_names
            st.session_state.model_loaded = True
            
            return model
        except Exception as e:
            st.error(f"Failed to save model: {str(e)}")
            return None

def predict_gesture(model, model_type, landmarks, confidence_threshold=0.7):
    """Make prediction only if confidence exceeds threshold"""
    try:
        if model_type == "SVM":
            proba = model.predict_proba([landmarks])[0]
            max_prob = np.max(proba)
            predicted_class = model.predict([landmarks])[0]
            if max_prob > confidence_threshold:
                return predicted_class, max_prob
        else:
            # Reshape for neural networks
            input_data = landmarks.reshape(1, landmarks.shape[0], 1)
            pred = model.predict(input_data, verbose=0)
            max_prob = np.max(pred)
            predicted_class = np.argmax(pred)
            if max_prob > confidence_threshold:
                return predicted_class, max_prob
        
        # Return None if confidence is too low
        return None, 0
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None, 0

# Update execute_action() to check confidence:
def execute_action(action_name, confidence):
    global last_action_time
    if confidence < 0.85:  # Only trigger if confident
        return
    # Rest of your existing code...
    
def load_saved_data():
    """Safely load saved data with comprehensive error handling"""
    default_data = {"gesture_data": {}, "mappings": {}}
    data_file = Path("gestures.json")
    
    if not data_file.exists():
        return {}, {}
    
    try:
        # Check if file is empty
        if data_file.stat().st_size == 0:
            return {}, {}
            
        with open(data_file, 'r') as f:
            data = json.load(f)
            
            # Validate data structure
            if not all(key in data for key in ["gesture_data", "mappings"]):
                st.warning("Invalid data format - resetting to defaults")
                return {}, {}
                
            # Convert loaded lists back to numpy arrays
            gesture_data = {}
            for gesture, samples in data.get("gesture_data", {}).items():
                gesture_data[gesture] = [np.array(sample) for sample in samples]
                
            return gesture_data, data.get("mappings", {})
            
    except (json.JSONDecodeError, ValueError) as e:
        st.warning(f"Corrupted data file: {str(e)} - resetting to defaults")
        return {}, {}
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return {}, {}

def save_data(gesture_data, mappings):
    """Atomic save operation with numpy array conversion"""
    temp_path = Path("gestures_temp.json")
    final_path = Path("gestures.json")
    
    try:
        # Convert numpy arrays to lists
        serializable_data = {
            "gesture_data": numpy_to_serializable(gesture_data),
            "mappings": mappings
        }
        
        # Write to temp file
        with open(temp_path, 'w') as f:
            json.dump(serializable_data, f, indent=4)
        
        # Create backup
        if final_path.exists():
            backup_path = final_path.with_suffix('.bak')
            final_path.replace(backup_path)
        
        # Atomic replace
        temp_path.replace(final_path)
        
    except Exception as e:
        st.error(f"Failed to save data: {str(e)}")
        if temp_path.exists():
            temp_path.unlink()
        raise

def execute_action(action_name):
    """Execute media action with better visual feedback"""
    global last_action_time
    
    current_time = time.time()
    if current_time - last_action_time < ACTION_COOLDOWN:
        return
    
    action_map = {
        "play_pause": ("⏯️ Play/Pause", lambda: pyautogui.press("playpause")),
        "volume_up": ("🔊 Volume Up", lambda: pyautogui.press("volumeup")),
        "volume_down": ("🔉 Volume Down", lambda: pyautogui.press("volumedown")),
        "next_track": ("⏭️ Next Track", lambda: pyautogui.press("nexttrack")),
        "prev_track": ("⏮️ Previous Track", lambda: pyautogui.press("prevtrack")),
        "mute": ("🔇 Mute", lambda: pyautogui.press("volumemute")),
        "fullscreen": ("🖥️ Fullscreen", lambda: pyautogui.press("f"))
    }
    
    if action_name in action_map:
        emoji, action_func = action_map[action_name]
        try:
            action_func()
            last_action_time = current_time
            st.toast(f"{emoji} {action_name.replace('_', ' ').title()}", icon="✅")
        except Exception as e:
            st.error(f"❌ Failed to execute {action_name}: {str(e)}")

def load_saved_model():
    """Load model from file if exists"""
    model_types = ["SVM", "CNN", "LSTM"]
    for model_type in model_types:
        model_path = f"gesture_model_{model_type}.h5" if model_type != "SVM" else f"gesture_model_{model_type}.pkl"
        if Path(model_path).exists():
            try:
                if model_type == "SVM":
                    model = joblib.load(model_path)
                else:
                    # Clear session and load with TensorFlow
                    tf.keras.backend.clear_session()
                    model = tf.keras.models.load_model(model_path)
                
                st.session_state.model = model
                st.session_state.model_type = model_type
                st.session_state.model_loaded = True
                return True
            except Exception as e:
                st.error(f"Error loading {model_type} model: {str(e)}")
                try:
                    # Attempt to remove corrupt file
                    os.remove(model_path)
                    st.warning(f"Removed potentially corrupt {model_type} model file")
                except Exception as cleanup_error:
                    st.error(f"Could not remove corrupt file: {str(cleanup_error)}")
    return False



def show_home_page():
    # Custom CSS for advanced styling with dark mode support
    st.markdown(f"""
    <style>
    .hero {{
        background: linear-gradient(135deg, {'#2a3f5f' if st.session_state.dark_mode else '#0061ff'}, 
                                          {'#4b6cb7' if st.session_state.dark_mode else '#60efff'});
        padding: 4rem 2rem;
        border-radius: 16px;
        color: white;
        text-align: center;
        margin-bottom: 3rem;
        box-shadow: 0 10px 20px rgba(0,0,0,0.1);
        animation: gradient 15s ease infinite;
        background-size: 400% 400%;
    }}
    @keyframes gradient {{
        0% {{background-position: 0% 50%}}
        50% {{background-position: 100% 50%}}
        100% {{background-position: 0% 50%}}
    }}
    .feature-card {{
        background: {'#1e2130' if st.session_state.dark_mode else 'white'};
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        height: 100%;
        border: 1px solid {'#2a3f5f' if st.session_state.dark_mode else '#e0e0e0'};
    }}
    .feature-card:hover {{
        transform: translateY(-5px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.12);
    }}
    .feature-card h3 {{
        color: {'#ffffff' if st.session_state.dark_mode else '#2a3f5f'};
    }}
    .feature-card p {{
        color: {'#b0b0b0' if st.session_state.dark_mode else '#666666'};
    }}
    .stats-container {{
        background: {'rgba(30, 33, 48, 0.9)' if st.session_state.dark_mode else 'rgba(255,255,255,0.9)'};
        backdrop-filter: blur(10px);
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        border: 1px solid {'#2a3f5f' if st.session_state.dark_mode else '#e0e0e0'};
    }}
    .tech-logo {{
        filter: {'invert(1)' if st.session_state.dark_mode else 'invert(0)'};
        transition: transform 0.3s ease;
    }}
    .tech-logo:hover {{
        transform: scale(1.1);
    }}
    </style>
    """, unsafe_allow_html=True)

    # Hero Section
    st.markdown("""
    <div class="hero">
        <h1 style="font-size: 3.5rem; margin-bottom: 1rem;">GestureVision Pro</h1>
        <p style="font-size: 1.5rem; margin-bottom: 2rem;">Next-Gen AI-Powered Gesture Recognition System</p>
        <div style="display: flex; justify-content: center; gap: 1rem;">
            <a href="#features" style="background: white; color: #0061ff; padding: 0.75rem 1.5rem; border-radius: 50px; text-decoration: none; font-weight: bold;">Explore Features</a>
            <a href="#demo" style="background: rgba(255,255,255,0.2); color: white; padding: 0.75rem 1.5rem; border-radius: 50px; text-decoration: none; font-weight: bold;">See Demo</a>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Introduction
    st.markdown("""
    ## Revolutionizing Human-Computer Interaction
    
    GestureVision Pro leverages cutting-edge machine learning and computer vision to transform 
    simple hand gestures into powerful commands. Our system offers:
    """)
    
    # Feature Highlights
    st.markdown('<a name="features"></a>', unsafe_allow_html=True)
    st.subheader("✨ Key Features")
    
    cols = st.columns(3)
    with cols[0]:
        with st.container():
            st.markdown("""
            <div class="feature-card">
                <h3>🎯 Precision Tracking</h3>
                <p>21-point hand landmark detection with sub-millimeter accuracy using MediaPipe's advanced algorithms</p>
                <div style="height: 120px; background: url('https://mediapipe.dev/images/mobile/hand_landmarks.png') center/contain no-repeat; filter: {dm_filter};"></div>
            </div>
            """.format(dm_filter="invert(1)" if st.session_state.dark_mode else "invert(0)"), unsafe_allow_html=True)
    
    with cols[1]:
        with st.container():
            st.markdown("""
            <div class="feature-card">
                <h3>🧠 Multi-Model Architecture</h3>
                <p>Advanced model comparison between SVM, CNN, LSTM and MobileNet architectures</p>
                <div style="height: 120px; background: url('https://miro.medium.com/max/1400/1*5hV2QbTF6Q1F7Vl7W4Qw1A.png') center/contain no-repeat; filter: {dm_filter};"></div>
            </div>
            """.format(dm_filter="invert(1)" if st.session_state.dark_mode else "invert(0)"), unsafe_allow_html=True)
    
    with cols[2]:
        with st.container():
            st.markdown("""
            <div class="feature-card">
                <h3>⚡ Real-Time Control</h3>
                <p>Low-latency processing with &lt;30ms response time for seamless interaction</p>
                <div style="height: 120px; background: url('https://developers.google.com/static/mediapipe/images/solutions/hand-landmarks.png') center/contain no-repeat; filter: {dm_filter};"></div>
            </div>
            """.format(dm_filter="invert(1)" if st.session_state.dark_mode else "invert(0)"), unsafe_allow_html=True)

    # Stats Dashboard
    if st.session_state.gesture_data or st.session_state.mappings:
        st.markdown("---")
        st.subheader("📈 Gesture Analytics Dashboard")
        
        with st.container():
            st.markdown("""
            <div class="stats-container">
                <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem;">
            """, unsafe_allow_html=True)
            
            cols = st.columns(4)
            metrics = [
                ("Gestures Recorded", len(st.session_state.gesture_data), "#4CAF50"),
                ("Total Samples", sum(len(v) for v in st.session_state.gesture_data.values()), "#2196F3"),
                ("Actions Mapped", len(st.session_state.mappings), "#FF9800"),
                ("Active Model", st.session_state.model_type if st.session_state.model else "None", "#9C27B0")
            ]
            
            for col, (label, value, color) in zip(cols, metrics):
                with col:
                    st.markdown(f"""
                    <div style="text-align: center;">
                        <div style="font-size: 1rem; color: {'#b0b0b0' if st.session_state.dark_mode else '#666'}; margin-bottom: 0.5rem;">{label}</div>
                        <div style="font-size: 2rem; font-weight: bold; color: {color};">{value}</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.markdown("</div></div>", unsafe_allow_html=True)

    # Quick Start Guide
    st.markdown("---")
    st.subheader("🚀 Getting Started")
    
    with st.expander("Step-by-Step Implementation Guide", expanded=True):
        steps = [
            ("1️⃣ Record Gestures", "Capture multiple samples of each gesture for robust training"),
            ("2️⃣ Train Model", "Select the optimal AI architecture for your use case"),
            ("3️⃣ Map Actions", "Assign system commands to each recognized gesture"),
            ("4️⃣ Deploy", "Run real-time recognition and control your environment")
        ]
        
        for step, description in steps:
            st.markdown(f"""
            <div style="display: flex; align-items: flex-start; margin-bottom: 1rem;">
                <div style="font-size: 1.5rem; margin-right: 1rem; color: {'#ffffff' if st.session_state.dark_mode else '#000000'};">{step.split(' ')[0]}</div>
                <div>
                    <h4 style="margin: 0 0 0.25rem 0; color: {'#ffffff' if st.session_state.dark_mode else '#2a3f5f'};">{step.split(' ')[1]}</h4>
                    <p style="margin: 0; color: {'#b0b0b0' if st.session_state.dark_mode else '#666'};">{description}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Demo Section
    st.markdown('<a name="demo"></a>', unsafe_allow_html=True)
    st.markdown("---")
    st.subheader("🎥 See It In Action")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.video("https://www.youtube.com/watch?v=example_gesture_demo")  # Replace with your demo video
    
    with col2:
        st.markdown(f"""
        ### Experience GestureVision
        <p style="color: {'#b0b0b0' if st.session_state.dark_mode else '#666'};">Watch how our technology transforms simple gestures into powerful commands:</p>
        
        <ul style="color: {'#b0b0b0' if st.session_state.dark_mode else '#666'};">
            <li>✋ Palm open → Play/Pause media</li>
            <li>👍 Thumbs up → Volume increase</li>
            <li>👎 Thumbs down → Volume decrease</li>
            <li>🤏 Pinch → Screenshot capture</li>
        </ul>
        
        <a href="#" style="color: {'#60efff' if st.session_state.dark_mode else '#0061ff'}; text-decoration: none; font-weight: bold;">Request enterprise demo →</a>
        """, unsafe_allow_html=True)

    # Technology Stack
    st.markdown("---")
    st.subheader("🛠️ Powered By")
    
    tech_cols = st.columns(5)
    tech_logos = [
        ("https://upload.wikimedia.org/wikipedia/commons/0/05/Scikit_learn_logo_small.svg", "Scikit-learn"),
        ("https://upload.wikimedia.org/wikipedia/commons/a/ae/TensorFlow_logo.svg", "TensorFlow"),
        ("https://upload.wikimedia.org/wikipedia/commons/3/38/MediaPipe_logo.png", "MediaPipe"),
        ("https://upload.wikimedia.org/wikipedia/commons/3/31/NumPy_logo_2020.svg", "NumPy"),
        ("https://streamlit.io/images/brand/streamlit-mark-color.png", "Streamlit")
    ]
    
    for col, (logo, name) in zip(tech_cols, tech_logos):
        with col:
            st.image(logo, width=80, output_format="PNG", use_column_width=False, 
                   caption=name, clamp=True, channels="RGB")
            st.markdown(f"<div style='text-align: center; color: {'#ffffff' if st.session_state.dark_mode else '#000000'};'>{name}</div>", 
                       unsafe_allow_html=True)

def show_record_page():
    """Professional gesture recording interface with enhanced UI"""
    # Custom CSS for dark/light mode compatibility
    st.markdown(f"""
    <style>
    .recording-card {{
        background: {'#1e2130' if st.session_state.dark_mode else '#ffffff'};
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid {'#2a3f5f' if st.session_state.dark_mode else '#e0e0e0'};
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        margin-bottom: 1rem;
    }}
    .gesture-card {{
        background: {'#1e2130' if st.session_state.dark_mode else '#ffffff'};
        border-radius: 10px;
        padding: 1.25rem;
        border: 1px solid {'#2a3f5f' if st.session_state.dark_mode else '#e0e0e0'};
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        transition: all 0.3s ease;
    }}
    .gesture-card:hover {{
        transform: translateY(-3px);
        box-shadow: 0 6px 16px rgba(0,0,0,0.1);
    }}
    .tab-content {{
        padding: 1rem 0;
    }}
    .progress-container {{
        background: {'#2a3f5f' if st.session_state.dark_mode else '#f0f2f6'};
        border-radius: 8px;
        padding: 0.5rem;
        margin-bottom: 1rem;
    }}
    </style>
    """, unsafe_allow_html=True)

    st.header("📸 Gesture Recording Studio")
    st.caption("Capture and manage your custom gesture dataset for training")

    # Initialize session state for recording status
    if 'recording_complete' not in st.session_state:
        st.session_state.recording_complete = False
    if 'current_gesture' not in st.session_state:
        st.session_state.current_gesture = None

    # Display current gestures in a card layout
    if st.session_state.gesture_data:
        with st.container():
            st.subheader("📁 Your Gesture Library")
            cols = st.columns(3)
            
            for i, (gesture, samples) in enumerate(st.session_state.gesture_data.items()):
                with cols[i % 3]:
                    with st.container():
                        st.markdown(f"""
                        <div class="gesture-card">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <h4 style="margin: 0; color: {'#ffffff' if st.session_state.dark_mode else '#2a3f5f'};">{gesture.replace('_', ' ').title()}</h4>
                                <span style="font-size: 0.9rem; color: {'#b0b0b0' if st.session_state.dark_mode else '#666'};">{len(samples)} samples</span>
                            </div>
                            <div style="margin-top: 0.5rem;">
                        """, unsafe_allow_html=True)
                        
                        if gesture in st.session_state.mappings:
                            st.success(f"🔗 Mapped to: {st.session_state.mappings[gesture].replace('_', ' ').title()}")
                        else:
                            st.warning("⚠️ Not mapped")
                            
                        st.markdown("</div></div>", unsafe_allow_html=True)
            st.markdown("---")

    # Main recording interface with tabs
    tab1, tab2 = st.tabs(["➕ Create New Gesture", "📥 Enhance Existing Gesture"])
    
    with tab1:
        with st.container():
            st.markdown("""
            <div class="tab-content">
                <h3 style="color: {'#ffffff' if st.session_state.dark_mode else '#2a3f5f'}; margin-bottom: 1rem;">Create New Gesture</h3>
            """, unsafe_allow_html=True)
            
            with st.form("new_gesture_form", clear_on_submit=True):
                with st.container():
                    st.markdown("""
                    <div class="recording-card">
                    """, unsafe_allow_html=True)
                    
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        gesture_name = st.text_input(
                            "✏️ Gesture Name (e.g., 'thumbs_up')", 
                            help="Use descriptive names without spaces",
                            key="new_gesture_name"
                        )
                    with col2:
                        num_samples = st.slider(
                            "Sample Count", 
                            20, 200, 50,
                            help="Recommended: 50-100 samples per gesture"
                        )
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    if st.form_submit_button("🎬 Start Recording Session", use_container_width=True):
                        if not gesture_name:
                            st.error("Please enter a valid gesture name")
                        elif gesture_name in st.session_state.gesture_data:
                            st.error(f"'{gesture_name}' already exists! Use the 'Enhance' tab")
                        else:
                            st.session_state.current_gesture = gesture_name
                            with st.spinner(f"🔴 Recording {num_samples} samples for '{gesture_name}'..."):
                                samples = record_gesture_samples(gesture_name, num_samples)
                                if samples:
                                    st.session_state.gesture_data[gesture_name] = samples
                                    save_data(st.session_state.gesture_data, st.session_state.mappings)
                                    st.session_state.recording_complete = True
                                    st.toast(f"✅ Successfully recorded {len(samples)} samples", icon="🎉")
                                    st.rerun()
            
            st.markdown("</div>", unsafe_allow_html=True)
    
    with tab2:
        with st.container():
            st.markdown("""
            <div class="tab-content">
                <h3 style="color: {'#ffffff' if st.session_state.dark_mode else '#2a3f5f'}; margin-bottom: 1rem;">Enhance Existing Gesture</h3>
            """, unsafe_allow_html=True)
            
            if not st.session_state.gesture_data:
                st.warning("No existing gestures found. Create your first gesture using the 'Create' tab.")
            else:
                with st.form("existing_gesture_form", clear_on_submit=True):
                    with st.container():
                        st.markdown("""
                        <div class="recording-card">
                        """, unsafe_allow_html=True)
                        
                        gesture_name = st.selectbox(
                            "Select Gesture to Enhance",
                            list(st.session_state.gesture_data.keys()),
                            help="Improve recognition by adding more samples"
                        )
                        
                        num_samples = st.slider(
                            "Additional Samples", 
                            10, 100, 20,
                            help="Add more samples to improve model accuracy"
                        )
                        
                        st.markdown("</div>", unsafe_allow_html=True)
                        
                        if st.form_submit_button("📸 Record Additional Samples", use_container_width=True):
                            st.session_state.current_gesture = gesture_name
                            with st.spinner(f"🔴 Recording {num_samples} additional samples..."):
                                new_samples = record_gesture_samples(gesture_name, num_samples)
                                if new_samples:
                                    st.session_state.gesture_data[gesture_name].extend(new_samples)
                                    save_data(st.session_state.gesture_data, st.session_state.mappings)
                                    st.session_state.recording_complete = True
                                    st.toast(f"✅ Added {len(new_samples)} samples (Total: {len(st.session_state.gesture_data[gesture_name])})", icon="👍")
                                    st.rerun()
            
            st.markdown("</div>", unsafe_allow_html=True)

    # Show "Add Another Gesture" button after successful recording
    if st.session_state.recording_complete and st.session_state.current_gesture:
        if st.button(f"➕ Record Another Gesture", type="primary", use_container_width=True):
            st.session_state.recording_complete = False
            st.session_state.current_gesture = None
            st.rerun()
    
    # Gesture management section
    if st.session_state.gesture_data:
        st.markdown("---")
        with st.container():
            st.subheader("⚙️ Dataset Management")
            
            with st.expander("🗂️ Gesture Administration", expanded=False):
                with st.container():
                    st.markdown("""
                    <div class="recording-card">
                    """, unsafe_allow_html=True)
                    
                    gesture_to_delete = st.selectbox(
                        "Select gesture to manage",
                        list(st.session_state.gesture_data.keys()),
                        key="delete_select"
                    )
                    
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        if st.button("⚠️ Delete Gesture", type="primary", use_container_width=True):
                            del st.session_state.gesture_data[gesture_to_delete]
                            if gesture_to_delete in st.session_state.mappings:
                                del st.session_state.mappings[gesture_to_delete]
                            save_data(st.session_state.gesture_data, st.session_state.mappings)
                            st.toast(f"🗑️ Deleted '{gesture_to_delete}'", icon="✅")
                            st.rerun()
                    with col2:
                        if st.button("📊 View Samples", use_container_width=True):
                            st.write(f"Sample data for {gesture_to_delete}:")
                            st.write(st.session_state.gesture_data[gesture_to_delete][:3])  # Show first 3 samples
                    
                    st.markdown("</div>", unsafe_allow_html=True)
def show_train_page():
    """Enhanced training page with MobileNetV2 pretrained model"""
    st.header("🎯 Train Gesture Model")
    
    # Initialize all variables at the start
    incremental = False
    freeze_base = True
    fine_tune_epochs = 0
    
    # Check if enough gestures are recorded
    if len(st.session_state.get('gesture_data', {})) < 2:
        with st.container(border=True):
            cols = st.columns([1, 4])
            with cols[0]:
                st.warning("⚠️")
            with cols[1]:
                st.markdown("**Not enough gestures recorded**")
                st.caption("Record at least 2 different gestures first")
                if st.button("Go to Record Page", key="go_to_record"):
                    st.session_state.page = "Record"
                    st.rerun()
        return
    
    # Check for unmapped gestures
    unmapped_gestures = [g for g in st.session_state.gesture_data if g not in st.session_state.get('mappings', {})]
    if unmapped_gestures:
        with st.container(border=True):
            cols = st.columns([1, 4])
            with cols[0]:
                st.warning("⚠️")
            with cols[1]:
                st.markdown(f"**{len(unmapped_gestures)} unmapped gestures**")
                st.caption("Map them in 'Map Gestures' page for full functionality")

    # ==================== MODEL SELECTION ====================
    st.subheader("1. Model Configuration")
    
    # Use tabs for better organization
    tab1, tab2 = st.tabs(["Model Selection", "Training Parameters"])
    
    with tab1:
        MODEL_OPTIONS = {
            "SVM": "Support Vector Machine (Fast)",
            "CNN": "Custom CNN (Balanced)",
            "MobileNet": "MobileNetV2 (Pretrained)",
            "LSTM": "Long Short-Term Memory"
        }
        
        model_type = st.radio(
            "Select Model Architecture",
            options=list(MODEL_OPTIONS.keys()),
            format_func=lambda x: MODEL_OPTIONS[x],
            horizontal=True
        )
        
        # Model info in a compact card
        with st.expander("📚 Model Details", expanded=True):
            if model_type == "SVM":
                st.markdown("""
                - **Best for**: Quick prototyping
                - **Speed**: ⚡ 1-2ms predictions
                - **Accuracy**: 85-92%
                - **Hardware**: CPU efficient
                """)
            elif model_type == "CNN":
                st.markdown("""
                - **Best for**: Hand shapes
                - **Speed**: ⏱️ 5-10ms
                - **Accuracy**: 90-95%
                - **Hardware**: Needs GPU
                """)
            elif model_type == "MobileNet":
                st.markdown("""
                - **Best for**: Mobile/Edge devices
                - **Speed**: ⚡ 3-5ms
                - **Accuracy**: 92-96%
                - **Hardware**: Optimized for all
                """)
            else:  # LSTM
                st.markdown("""
                - **Best for**: Motion patterns
                - **Speed**: 🐢 15-30ms
                - **Accuracy**: 92-97%
                - **Hardware**: Needs GPU
                """)
            
            # Complexity meter
            complexity = {
                "SVM": 20,
                "CNN": 60,
                "MobileNet": 50, 
                "LSTM": 80
            }
            st.progress(complexity.get(model_type, 50), 
                       text=f"{'Low' if complexity.get(model_type, 0) < 40 else 'Medium' if complexity.get(model_type, 0) < 70 else 'High'} complexity")

    with tab2:
        if model_type != "SVM":
            cols = st.columns(2)
            with cols[0]:
                epochs = st.slider("Training Epochs", 5, 100, 30)
            with cols[1]:
                batch_size = st.selectbox("Batch Size", [16, 32, 64], index=1)
            
            if model_type == "MobileNet":
                freeze_base = st.checkbox("Freeze Base Layers", value=True)
                if not freeze_base:
                    fine_tune_epochs = st.slider("Fine-tuning Epochs", 1, 20, 5)
            
            if st.session_state.model and st.session_state.model_type == model_type:
                incremental = st.checkbox("Continue Training Existing Model", value=True)
            else:
                incremental = False

    # ==================== TRAINING VISUALIZATION ====================
    st.subheader("2. Training Progress")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        with st.container(border=True):
            st.markdown("**Live Metrics**")
            metric1 = st.empty()  # Training Accuracy
            metric2 = st.empty()  # Validation Loss
            progress_bar = st.progress(0, text="Ready to train")
    
    with col2:
        chart_placeholder = st.empty()
        confusion_placeholder = st.empty()

    # ==================== TRAINING CALLBACK ====================
    class TrainingCallback(tf.keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.history = {'accuracy': [], 'loss': [], 'val_accuracy': [], 'val_loss': []}
            
        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            
            # Store metrics
            for k, v in logs.items():
                if k in self.history:
                    self.history[k].append(v)
            
            # Update progress
            if 'epochs' in locals() and epochs:
                progress = (epoch + 1) / epochs
                progress_bar.progress(min(progress, 1.0), text=f"Epoch {epoch+1}/{epochs}")
            
            # Update metrics
            metric1.metric("Training Accuracy", f"{logs.get('accuracy', 0):.2%}")
            metric2.metric("Validation Loss", f"{logs.get('val_loss', 0):.4f}")
            
            # Update charts
            if len(self.history['accuracy']) > 1:
                with col2:
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
                    
                    # Accuracy plot
                    ax1.plot(self.history['accuracy'], label='Train')
                    if self.history['val_accuracy']:
                        ax1.plot(self.history['val_accuracy'], label='Validation')
                    ax1.set_title('Model Accuracy')
                    ax1.set_ylabel('Accuracy')
                    ax1.legend(loc='lower right')
                    
                    # Loss plot
                    ax2.plot(self.history['loss'], label='Train')
                    if self.history['val_loss']:
                        ax2.plot(self.history['val_loss'], label='Validation')
                    ax2.set_title('Model Loss')
                    ax2.set_ylabel('Loss')
                    ax2.legend(loc='upper right')
                    
                    chart_placeholder.pyplot(fig)
                    plt.close(fig)

    # ==================== FIXED MOBILENET IMPLEMENTATION ====================
    def create_mobilenet_model(input_shape, num_classes, freeze_base=True):
        """Fixed implementation for 1D gesture data"""
        inputs = tf.keras.Input(shape=input_shape)
        
        # First expand to 2D with single channel
        x = tf.keras.layers.Reshape((input_shape[0], 1))(inputs)
        
        # Standardize input length to 224 (MobileNet's default)
        target_length = 224
        if input_shape[0] < target_length:
            # Pad with zeros if too short
            x = tf.keras.layers.ZeroPadding1D((0, target_length - input_shape[0]))(x)
        elif input_shape[0] > target_length:
            # Trim if too long
            x = tf.keras.layers.Cropping1D((0, input_shape[0] - target_length))(x)
        
        # Convert to 3 channels by repeating the data
        x = tf.keras.layers.Concatenate(axis=-1)([x, x, x])
        
        # Reshape to (224, 224, 3) by repeating rows
        x = tf.keras.layers.Lambda(lambda x: tf.repeat(tf.expand_dims(x, axis=1), repeats=224, axis=1))(x)
        
        # Load MobileNetV2 with proper input shape
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(target_length, target_length, 3),
            include_top=False,
            weights='imagenet',
            pooling='avg'
        )
        base_model.trainable = not freeze_base
        
        x = base_model(x)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
        
        model = tf.keras.Model(inputs, outputs)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    # ==================== TRAINING EXECUTION ====================
    if st.button("🚀 Start Training", type="primary", use_container_width=True):
        if model_type != "SVM" and any(len(samples) < 20 for samples in st.session_state.gesture_data.values()):
            with st.container(border=True):
                st.error("⚠️ Low Sample Warning")
                st.markdown("Some gestures have <20 samples - accuracy may suffer")
                if not st.checkbox("Train anyway (not recommended)"):
                    return

        with st.spinner(f"Training {MODEL_OPTIONS[model_type]} model..."):
            try:
                # Prepare data
                X, y = [], []
                gesture_names = list(st.session_state.gesture_data.keys())
                min_samples = min(len(samples) for samples in st.session_state.gesture_data.values())
                
                for gesture_name, samples in st.session_state.gesture_data.items():
                    X.extend(samples[:min_samples])
                    y.extend([gesture_names.index(gesture_name)] * min_samples)
                
                X = np.array(X)
                y = np.array(y)
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, 
                    test_size=0.2, 
                    stratify=y,
                    random_state=42
                )
                
                if model_type == "SVM":
                    progress_bar.progress(0.3, text="Scaling features...")
                    model = make_pipeline(
                        StandardScaler(), 
                        SVC(kernel='rbf', C=10, gamma='scale', probability=True)
                    )
                    progress_bar.progress(0.6, text="Training SVM...")
                    model.fit(X_train, y_train)
                    progress_bar.progress(1.0, text="Training complete!")
                    
                    # Evaluate SVM
                    y_pred = model.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred)
                    
                    # Show results
                    st.success(f"✅ SVM trained successfully! Accuracy: {accuracy*100:.1f}%")
                    with col2:
                        st.subheader("Classification Report")
                        report = classification_report(y_test, y_pred, target_names=gesture_names, output_dict=True)
                        st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)
                    
                else:  # Neural networks
                    # Prepare data for NN
                    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
                    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
                    y_train = to_categorical(y_train, len(gesture_names))
                    y_test = to_categorical(y_test, len(gesture_names))
                    
                    # Create or load model
                    if incremental and st.session_state.model_type == model_type:
                        model = st.session_state.model
                        st.info("Continuing training with existing model...")
                        tf.keras.backend.set_value(model.optimizer.learning_rate, 0.0001)
                    else:
                        if model_type == "CNN":
                            model = Sequential([
                                Conv1D(64, 3, activation='relu', input_shape=(X_train.shape[1], 1)),
                                MaxPooling1D(2),
                                Conv1D(128, 3, activation='relu'),
                                MaxPooling1D(2),
                                Flatten(),
                                Dense(128, activation='relu'),
                                Dropout(0.5),
                                Dense(len(gesture_names), activation='softmax')
                            ])
                        elif model_type == "MobileNet":
                            model = create_mobilenet_model(
                                input_shape=(X_train.shape[1], 1),
                                num_classes=len(gesture_names),
                                freeze_base=freeze_base
                            )
                        else:  # LSTM
                            model = Sequential([
                                LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], 1)),
                                LSTM(128),
                                Dense(64, activation='relu'),
                                Dense(len(gesture_names), activation='softmax')
                            ])
                        
                        model.compile(
                            optimizer='adam', 
                            loss='categorical_crossentropy', 
                            metrics=['accuracy']
                        )
                    
                    # Initialize progress bar
                    progress_bar.progress(0, text="Starting training...")
                    
                    # Train model
                    history = model.fit(
                        X_train, y_train,
                        validation_data=(X_test, y_test),
                        epochs=epochs,
                        batch_size=batch_size,
                        callbacks=[TrainingCallback()],
                        verbose=0
                    )
                    
                    # Fine-tuning if enabled
                    if model_type == "MobileNet" and not freeze_base and fine_tune_epochs > 0:
                        model.layers[5].trainable = True  # Unfreeze MobileNet
                        model.compile(
                            optimizer=tf.keras.optimizers.Adam(1e-5),
                            loss='categorical_crossentropy',
                            metrics=['accuracy']
                        )
                        history_fine = model.fit(
                            X_train, y_train,
                            validation_data=(X_test, y_test),
                            epochs=fine_tune_epochs,
                            batch_size=batch_size,
                            verbose=0
                        )
                        # Combine histories
                        for k in history.history:
                            history.history[k] += history_fine.history[k]
                    
                    # Final evaluation
                    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
                    st.success(f"✅ {model_type} trained successfully! Final Accuracy: {accuracy*100:.1f}%")
                
                # Confusion matrix
                with col2:
                    st.subheader("Confusion Matrix")
                    y_pred = model.predict(X_test)
                    if model_type != "SVM":
                        y_pred = np.argmax(y_pred, axis=1)
                        y_test = np.argmax(y_test, axis=1)
                    
                    fig, ax = plt.subplots(figsize=(8, 6))
                    cm = confusion_matrix(y_test, y_pred)
                    disp = ConfusionMatrixDisplay(
                        confusion_matrix=cm,
                        display_labels=gesture_names
                    )
                    disp.plot(ax=ax, cmap='Blues', values_format='d')
                    plt.xticks(rotation=45)
                    confusion_placeholder.pyplot(fig)
                    plt.close(fig)
                
                # Save model
                model_path = f"gesture_model_{model_type}.h5" if model_type != "SVM" else f"gesture_model_{model_type}.pkl"
                try:
                    if model_type == "SVM":
                        joblib.dump(model, model_path)
                    else:
                        # Remove existing file if it exists
                        if os.path.exists(model_path):
                            os.remove(model_path)
                        model.save(model_path)
                    
                    st.session_state.model = model
                    st.session_state.model_type = model_type
                    st.session_state.model_loaded = True
                    st.session_state.gesture_names = gesture_names
                    
                    st.toast("🎉 Model trained successfully!", icon="✅")
                except Exception as e:
                    st.error(f"Failed to save model: {str(e)}")
                    progress_bar.progress(0, text="Model save failed")
                
            except Exception as e:
                st.error(f"Training failed: {str(e)}")
                progress_bar.progress(0, text="Training failed")
                st.exception(e)
# Add this new function to display the comparison page
def show_comparison_page():
    """Advanced model comparison with interactive features"""
    st.header("📊 Advanced Model Comparison Dashboard")
    
    # Check available models
    available_models = []
    model_sizes = {}
    for model_type in MODEL_OPTIONS.keys():
        model_path = f"gesture_model_{model_type}.h5" if model_type != "SVM" else f"gesture_model_{model_type}.pkl"
        if Path(model_path).exists():
            available_models.append(model_type)
            model_sizes[model_type] = os.path.getsize(model_path) / (1024 * 1024)  # MB
    
    if len(available_models) < 2:
        st.warning("You need at least 2 trained models to compare")
        st.info("Go to the Train page to train different model types")
        if st.button("Go to Train Page"):
            st.session_state.page = "Train"
        return
    
    # Introduction with expandable info
    with st.expander("ℹ️ About Model Comparison", expanded=True):
        st.markdown("""
        This dashboard provides comprehensive comparison of your trained models across multiple dimensions:
        - **Accuracy Metrics**: Precision, Recall, F1-score per class
        - **Performance**: Prediction latency, throughput
        - **Resource Usage**: Memory footprint, hardware utilization
        - **Robustness**: Noise sensitivity, confidence distributions
        """)
    
    # Model selection with filters
    st.subheader("1️⃣ Model Selection")
    col1, col2 = st.columns(2)
    with col1:
        selected_models = st.multiselect(
            "Select models to compare (max 4 recommended)",
            options=available_models,
            default=available_models[:min(4, len(available_models))],
            format_func=lambda x: f"{MODEL_OPTIONS[x]} ({model_sizes[x]:.1f}MB)"
        )
    with col2:
        test_size = st.slider("Test set percentage", 10, 50, 20)
        num_test_samples = st.slider("Max test samples per model", 100, 2000, 500)
    
    if not selected_models or len(selected_models) < 2:
        st.warning("Please select at least 2 models")
        return
    
    # Load models with progress
    models = {}
    with st.spinner(f"Loading {len(selected_models)} models..."):
        progress_bar = st.progress(0)
        for i, model_type in enumerate(selected_models):
            try:
                model_path = f"gesture_model_{model_type}.h5" if model_type != "SVM" else f"gesture_model_{model_type}.pkl"
                if model_type == "SVM":
                    model = joblib.load(model_path)
                else:
                    with tf.device('/cpu:0'):  # Load on CPU for consistent comparison
                        model = tf.keras.models.load_model(model_path)
                models[model_type] = model
                progress_bar.progress((i + 1) / len(selected_models))
            except Exception as e:
                st.error(f"Failed to load {model_type}: {str(e)}")
                continue
    
    if len(models) < 2:
        st.error("Couldn't load enough models")
        return
    
    # Prepare test data with stratification
    gesture_names = list(st.session_state.gesture_data.keys())
    X, y = [], []
    min_samples = min(len(samples) for samples in st.session_state.gesture_data.values())
    
    for gesture_name, samples in st.session_state.gesture_data.items():
        X.extend(samples[:min_samples])
        y.extend([gesture_names.index(gesture_name)] * min_samples)
    
    X = np.array(X)
    y = np.array(y)
    
    # Split data with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size/100, 
        random_state=42,
        stratify=y
    )
    
    # Limit test samples
    if len(X_test) > num_test_samples:
        X_test = X_test[:num_test_samples]
        y_test = y_test[:num_test_samples]
    
    # Prepare data for neural networks
    if any(m in selected_models for m in ["CNN", "LSTM", "MobileNet"]):
        X_test_nn = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
        y_test_nn = to_categorical(y_test, len(gesture_names))
    
    # ==================== EVALUATION METRICS ====================
    st.subheader("2️⃣ Performance Metrics")
    
    # Metrics storage
    metrics = {
        'Model': [],
        'Accuracy': [],
        'Precision (macro)': [],
        'Recall (macro)': [],
        'F1 (macro)': [],
        'Prediction Time (ms)': [],
        'Throughput (preds/sec)': [],
        'Model Size (MB)': [],
        'Parameters': [],
        'Memory Usage (MB)': []
    }
    
    # Class-wise metrics storage
    class_metrics = {
        model_type: {name: {'precision': [], 'recall': [], 'f1': []} 
                   for name in gesture_names}
        for model_type in selected_models
    }
    
    # Confidence distributions
    confidence_data = {model_type: [] for model_type in selected_models}
    
    # Evaluate each model
    with st.spinner("Running comprehensive evaluation..."):
        progress_text = st.empty()
        progress_bar = st.progress(0)
        
        for i, (model_type, model) in enumerate(models.items()):
            progress_text.text(f"Evaluating {MODEL_OPTIONS[model_type]}...")
            
            # Timing and prediction
            start_time = time.time()
            
            if model_type == "SVM":
                # SVM evaluation
                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                report = classification_report(y_test, y_pred, output_dict=True)
                
                # Store max confidence
                confidence_data[model_type] = np.max(y_proba, axis=1)
            else:
                # Neural network evaluation
                y_pred = np.argmax(model.predict(X_test_nn, verbose=0), axis=1)
                y_proba = model.predict(X_test_nn, verbose=0)
                accuracy = accuracy_score(y_test, y_pred)
                report = classification_report(y_test, y_pred, output_dict=True)
                
                # Store max confidence
                confidence_data[model_type] = np.max(y_proba, axis=1)
            
            # Timing
            elapsed = time.time() - start_time
            pred_time = (elapsed / len(X_test)) * 1000  # ms per prediction
            throughput = len(X_test) / elapsed  # preds/sec
            
            # Memory measurement
            if model_type == "SVM":
                mem_usage = model_sizes[model_type]  # Approximate
                params = "N/A"
            else:
                mem_usage = model_sizes[model_type]
                params = model.count_params()
            
            # Store metrics
            metrics['Model'].append(MODEL_OPTIONS[model_type])
            metrics['Accuracy'].append(accuracy)
            metrics['Precision (macro)'].append(report['macro avg']['precision'])
            metrics['Recall (macro)'].append(report['macro avg']['recall'])
            metrics['F1 (macro)'].append(report['macro avg']['f1-score'])
            metrics['Prediction Time (ms)'].append(pred_time)
            metrics['Throughput (preds/sec)'].append(throughput)
            metrics['Model Size (MB)'].append(model_sizes[model_type])
            metrics['Parameters'].append(params)
            metrics['Memory Usage (MB)'].append(mem_usage)
            
            # Store class-wise metrics
            for name in gesture_names:
                idx = gesture_names.index(name)
                if str(idx) in report:
                    class_metrics[model_type][name]['precision'].append(report[str(idx)]['precision'])
                    class_metrics[model_type][name]['recall'].append(report[str(idx)]['recall'])
                    class_metrics[model_type][name]['f1'].append(report[str(idx)]['f1-score'])
            
            progress_bar.progress((i + 1) / len(models))
    
    # ==================== INTERACTIVE VISUALIZATIONS ====================
    st.subheader("3️⃣ Visualization Dashboard")
    
    # Tabs for different visualization types
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Summary Metrics", 
        "🎯 Class Performance", 
        "⏱️ Speed Analysis",
        "📊 Confidence Analysis"
    ])
    
    with tab1:
        # Convert metrics to DataFrame
        df = pd.DataFrame(metrics)
        
        # Interactive Altair chart - Radar plot for comparison
        st.markdown("**Radar Chart Comparison**")
        radar_df = df.melt(id_vars=['Model'], 
                          value_vars=['Accuracy', 'Precision (macro)', 'Recall (macro)', 'F1 (macro)'],
                          var_name='Metric', value_name='Value')
        
        radar_chart = alt.Chart(radar_df).mark_line().encode(
            x='Metric:N',
            y=alt.Y('Value:Q', scale=alt.Scale(domain=[0, 1])),
            color='Model:N',
            tooltip=['Model', 'Metric', alt.Tooltip('Value', format='.2%')]
        ).properties(
            width=600,
            height=400
        ).interactive()
        
        st.altair_chart(radar_chart, use_container_width=True)
        
        # Metrics table with sorting
        st.markdown("**Detailed Metrics Table**")
        st.dataframe(
            df.style.format({
                'Accuracy': '{:.2%}',
                'Precision (macro)': '{:.2%}',
                'Recall (macro)': '{:.2%}',
                'F1 (macro)': '{:.2%}',
                'Prediction Time (ms)': '{:.2f}',
                'Throughput (preds/sec)': '{:.1f}',
                'Model Size (MB)': '{:.1f}',
                'Memory Usage (MB)': '{:.1f}'
            }).background_gradient(cmap='Blues', subset=['Accuracy', 'F1 (macro)']),
            use_container_width=True,
            height=400
        )
    
    with tab2:
        # Class-wise performance heatmap
        st.markdown("**Class-wise F1 Scores**")
        
        # Prepare data for heatmap
        heatmap_data = []
        for model_type in selected_models:
            for gesture in gesture_names:
                if class_metrics[model_type][gesture]['f1']:  # Check if data exists
                    heatmap_data.append({
                        'Model': MODEL_OPTIONS[model_type],
                        'Gesture': gesture,
                        'F1 Score': class_metrics[model_type][gesture]['f1'][0]
                    })
        
        if heatmap_data:
            heatmap_df = pd.DataFrame(heatmap_data)
            
            # Pivot for heatmap
            pivot_df = heatmap_df.pivot(index='Model', columns='Gesture', values='F1 Score')
            
            # Create heatmap
            plt.figure(figsize=(10, 6))
            sns.heatmap(pivot_df, annot=True, fmt=".2f", cmap="YlGnBu", vmin=0, vmax=1)
            plt.title("F1 Scores by Gesture and Model")
            plt.xticks(rotation=45)
            st.pyplot(plt)
            
            # Detailed class metrics expander
            with st.expander("View Detailed Class Metrics"):
                selected_gesture = st.selectbox("Select gesture", gesture_names)
                
                # Prepare data for the selected gesture
                gesture_data = []
                for model_type in selected_models:
                    if class_metrics[model_type][selected_gesture]['f1']:
                        gesture_data.append({
                            'Model': MODEL_OPTIONS[model_type],
                            'Precision': class_metrics[model_type][selected_gesture]['precision'][0],
                            'Recall': class_metrics[model_type][selected_gesture]['recall'][0],
                            'F1': class_metrics[model_type][selected_gesture]['f1'][0]
                        })
                
                if gesture_data:
                    gesture_df = pd.DataFrame(gesture_data)
                    st.dataframe(
                        gesture_df.style.format({
                            'Precision': '{:.2%}',
                            'Recall': '{:.2%}',
                            'F1': '{:.2%}'
                        }),
                        use_container_width=True
                    )
        else:
            st.warning("No class-wise metrics available")
    
    with tab3:
        # Speed vs Accuracy tradeoff
        st.markdown("**Speed-Accuracy Tradeoff**")
        
        speed_df = pd.DataFrame({
            'Model': df['Model'],
            'Accuracy': df['Accuracy'],
            'Prediction Time (ms)': df['Prediction Time (ms)'],
            'Model Size (MB)': df['Model Size (MB)']
        })
        
        # Create bubble chart
        bubble_chart = alt.Chart(speed_df).mark_circle(size=100).encode(
            x='Prediction Time (ms)',
            y='Accuracy',
            size='Model Size (MB)',
            color='Model',
            tooltip=['Model', 'Accuracy', 'Prediction Time (ms)', 'Model Size (MB)']
        ).properties(
            width=600,
            height=400
        ).interactive()
        
        st.altair_chart(bubble_chart, use_container_width=True)
        
        # Throughput comparison
        st.markdown("**Throughput Comparison**")
        throughput_bar = alt.Chart(df).mark_bar().encode(
            x='Model',
            y='Throughput (preds/sec)',
            color='Model',
            tooltip=['Model', 'Throughput (preds/sec)']
        )
        st.altair_chart(throughput_bar, use_container_width=True)
    
    with tab4:
        # Confidence distribution analysis
        st.markdown("**Prediction Confidence Distributions**")
        
        # Prepare confidence data
        conf_data = []
        for model_type, confs in confidence_data.items():
            for val in confs:
                conf_data.append({
                    'Model': MODEL_OPTIONS[model_type],
                    'Confidence': val
                })
        
        if conf_data:
            conf_df = pd.DataFrame(conf_data)
            
            # Violin plot
            plt.figure(figsize=(10, 6))
            sns.violinplot(data=conf_df, x='Model', y='Confidence', palette="Set2")
            plt.axhline(0.85, color='red', linestyle='--', label='Confidence Threshold')
            plt.title("Prediction Confidence Distribution")
            plt.xticks(rotation=45)
            plt.legend()
            st.pyplot(plt)
            
            # Confidence statistics
            st.markdown("**Confidence Statistics**")
            conf_stats = conf_df.groupby('Model')['Confidence'].agg(['mean', 'std', 'min', 'max'])
            st.dataframe(
                conf_stats.style.format('{:.2f}'),
                use_container_width=True
            )
        else:
            st.warning("No confidence data available")
    
    # ==================== RECOMMENDATION ENGINE ====================
    st.subheader("4️⃣ Smart Recommendations")
    
    if len(df) >= 2:
        # Find best in each category
        best_acc = df.loc[df['Accuracy'].idxmax()]
        fastest = df.loc[df['Prediction Time (ms)'].idxmin()]
        smallest = df.loc[df['Model Size (MB)'].idxmin()]
        best_balanced = df.loc[((df['Accuracy'] * 0.7) + (1/df['Prediction Time (ms)']) * 0.3).idxmax()]
        
        # Display recommendations
        cols = st.columns(4)
        with cols[0]:
            with st.container(border=True):
                st.markdown("**🥇 Best Accuracy**")
                st.metric("Model", best_acc['Model'])
                st.metric("Accuracy", f"{best_acc['Accuracy']:.1%}")
                st.metric("Speed", f"{best_acc['Prediction Time (ms)']:.1f} ms")
        
        with cols[1]:
            with st.container(border=True):
                st.markdown("**⚡ Fastest**")
                st.metric("Model", fastest['Model'])
                st.metric("Speed", f"{fastest['Prediction Time (ms)']:.1f} ms")
                st.metric("Accuracy", f"{fastest['Accuracy']:.1%}")
        
        with cols[2]:
            with st.container(border=True):
                st.markdown("**📦 Smallest**")
                st.metric("Model", smallest['Model'])
                st.metric("Size", f"{smallest['Model Size (MB)']:.1f} MB")
                st.metric("Accuracy", f"{smallest['Accuracy']:.1%}")
        
        with cols[3]:
            with st.container(border=True):
                st.markdown("**⚖️ Best Balanced**")
                st.metric("Model", best_balanced['Model'])
                st.metric("Score", f"{(best_balanced['Accuracy'] * 0.7 + (1/best_balanced['Prediction Time (ms)']) * 0.3):.2f}")
                st.metric("Acc/Speed", f"{best_balanced['Accuracy']:.1%}/{best_balanced['Prediction Time (ms)']:.1f}ms")
        
        # Use case recommendations
        st.markdown("**🔍 Recommended by Use Case**")
        
        use_cases = {
            "Real-time Application": fastest['Model'],
            "High Accuracy Needed": best_acc['Model'],
            "Mobile/Edge Deployment": smallest['Model'],
            "General Purpose": best_balanced['Model']
        }
        
        st.table(pd.DataFrame.from_dict(use_cases, orient='index', columns=['Recommended Model']))
    
    # ==================== EXPORT OPTIONS ====================
    st.subheader("5️⃣ Export Options")
    
    col1, col2 = st.columns(2)
    with col1:
        # Export metrics
        if st.button("📊 Export Metrics to CSV"):
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="model_comparison_metrics.csv",
                mime="text/csv"
            )
    
    with col2:
        # Export visualizations
        if st.button("📸 Export Dashboard as PDF"):
            # This would require additional libraries like pdfkit
            st.warning("PDF export requires server-side setup with pdfkit/wkhtmltopdf")
    
    # ==================== MODEL DEPLOYMENT CENTER ====================
    st.markdown("---")
    st.markdown("## 🚀 Model Deployment Center")
    
    # Display current model info
    current_model_type = st.session_state.get('model_type', None)
    current_model_name = MODEL_OPTIONS.get(current_model_type, "None")
    
    with st.container(border=True):
        cols = st.columns([1, 3])
        with cols[0]:
            st.metric("Current Active Model", current_model_name)
        with cols[1]:
            if current_model_type:
                st.success(f"✅ {current_model_name} is ready for live control")
            else:
                st.warning("⚠️ No model currently active for live control")
    
    # Model switching interface
    st.markdown("### 🔄 Switch Active Model")
    st.markdown("Select a different model to use for live gesture control:")
    
    if not available_models:
        st.error("No trained models found! Please train at least one model first.")
    else:
        # Create a select box with model details
        selected_model = st.selectbox(
            "Choose model to activate:",
            options=available_models,
            format_func=lambda x: f"{MODEL_OPTIONS[x]} ({'Neural Network' if x != 'SVM' else 'Machine Learning'})",
            help="Select which model to use for live gesture recognition"
        )
        
        # Display model details before switching
        if selected_model:
            model_details = {
                "CNN": "Best for: Hand shape recognition\nSpeed: Medium\nAccuracy: High",
                "LSTM": "Best for: Sequential gesture patterns\nSpeed: Slow\nAccuracy: Very High",
                "SVM": "Best for: Quick prototyping\nSpeed: Fast\nAccuracy: Medium",
                "MobileNet": "Best for: Mobile/Edge devices\nSpeed: Fast\nAccuracy: High"
            }
            
            with st.expander(f"🔍 {MODEL_OPTIONS[selected_model]} Details"):
                st.markdown(model_details.get(selected_model, "No details available"))
                
                # Show sample performance metrics if available
                if 'df' in locals():
                    model_metrics = df[df['Model'] == MODEL_OPTIONS[selected_model]]
                    if not model_metrics.empty:
                        st.markdown("**Performance Metrics:**")
                        
                        # Make a copy and drop Model column if it exists
                        display_df = model_metrics.copy()
                        if 'Model' in display_df.columns:
                            display_df = display_df.drop(columns=['Model'])
                        
                        # Transpose the DataFrame
                        transposed_df = display_df.T
                        
                        # Identify which numeric columns exist in the transposed DataFrame
                        existing_numeric_cols = [
                            col for col in transposed_df.columns 
                            if col in transposed_df.select_dtypes(include=[np.number]).columns
                        ]
                        
                        # Apply formatting only to existing numeric columns
                        if existing_numeric_cols:
                            styled_df = transposed_df.style.format("{:.2f}", subset=existing_numeric_cols)
                            st.dataframe(styled_df)
                        else:
                            st.dataframe(transposed_df)
                
        # Confirmation button with safety check
        if st.button("🔄 Switch to Selected Model", type="primary"):
            try:
                # Load the selected model
                model_path = f"gesture_model_{selected_model}.h5" if selected_model != "SVM" else f"gesture_model_{selected_model}.pkl"
                
                with st.spinner(f"Loading {MODEL_OPTIONS[selected_model]}..."):
                    if selected_model == "SVM":
                        new_model = joblib.load(model_path)
                    else:
                        new_model = tf.keras.models.load_model(model_path)
                    
                    # Update session state
                    st.session_state.model = new_model
                    st.session_state.model_type = selected_model
                    st.session_state.model_loaded = True
                
                st.success(f"✅ Successfully activated {MODEL_OPTIONS[selected_model]}!")
                st.balloons()
                
                # Show quick actions
                st.markdown("### Next Steps:")
                cols = st.columns(2)
                with cols[0]:
                    if st.button("🎮 Test in Live Control", help="Go to live control page"):
                        st.session_state.page = "Control"
                        st.rerun()
                with cols[1]:
                    if st.button("📊 View Model Details", help="See detailed metrics"):
                        st.session_state.page = "Compare"
                        st.rerun()
                        
            except Exception as e:
                st.error(f"❌ Failed to switch models: {str(e)}")
                st.exception(e)
    
    # ==================== MODEL MANAGEMENT ====================
    st.markdown("---")
    st.markdown("## ⚙️ Model Management")
    
    with st.expander("🧹 Model Cleanup Tools", expanded=False):
        st.warning("These actions are irreversible!")
        
        cols = st.columns(3)
        with cols[0]:
            if st.button("🗑️ Delete All Models"):
                if st.checkbox("I understand this will delete ALL trained models"):
                    deleted = 0
                    for model_type in MODEL_OPTIONS.keys():
                        model_path = f"gesture_model_{model_type}.h5" if model_type != "SVM" else f"gesture_model_{model_type}.pkl"
                        if Path(model_path).exists():
                            try:
                                os.remove(model_path)
                                deleted += 1
                            except:
                                pass
                    st.session_state.model = None
                    st.session_state.model_type = None
                    st.info(f"Deleted {deleted} model files")
                    st.rerun()
        
        with cols[1]:
            if st.button("🔄 Reset Current Model"):
                st.session_state.model = None
                st.session_state.model_type = None
                st.success("Active model cleared")
        
        with cols[2]:
            if st.button("🔄 Refresh Model List"):
                st.rerun()


def show_mapping_page():
    """Enhanced mapping page with better visualization"""
    st.header("🔄 Map Gestures to Actions")
    
    if not st.session_state.gesture_data:
        st.warning("No gestures recorded yet!")
        if st.button("Go to Record Gestures"):
            st.session_state.page = "Record"
        return
    
    # Current mappings display
    st.subheader("📋 Current Mappings")
    
    if st.session_state.mappings:
        cols = st.columns(3)
        for i, (gesture, action) in enumerate(st.session_state.mappings.items()):
            with cols[i % 3]:
                with st.container(border=True):
                    st.markdown(f"**{gesture.replace('_', ' ').title()}**")
                    st.markdown(f"→ `{action.replace('_', ' ').title()}`")
                    if st.button("❌ Remove", key=f"remove_{gesture}", use_container_width=True):
                        del st.session_state.mappings[gesture]
                        save_data(st.session_state.gesture_data, st.session_state.mappings)
                        st.rerun()
    else:
        st.warning("No mappings yet - create your first one below!")
    
    st.markdown("---")
    
    # Create new mapping section
    st.subheader("➕ Create New Mapping")
    
    col1, col2 = st.columns(2)
    with col1:
        gesture_name = st.selectbox(
            "Select Gesture",
            sorted(list(st.session_state.gesture_data.keys())),
            help="Choose a recorded gesture"
        )
    with col2:
        action_name = st.selectbox(
            "Select Action",
            [
                "play_pause", "volume_up", "volume_down",
                "next_track", "prev_track", "mute", "fullscreen"
            ],
            format_func=lambda x: x.replace('_', ' ').title(),
            help="Choose an action to map"
        )
    
    if st.button("💾 Save Mapping", type="primary", use_container_width=True):
        st.session_state.mappings[gesture_name] = action_name
        save_data(st.session_state.gesture_data, st.session_state.mappings)
        st.success(f"Saved: {gesture_name} → {action_name}")
        st.rerun()
    
    # Quick actions section
    st.markdown("---")
    st.subheader("⚡ Quick Actions")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔍 View All Gestures", use_container_width=True):
            st.session_state.page = "Record"
    with col2:
        if st.button("🎮 Test in Live Control", use_container_width=True):
            st.session_state.page = "Control"


def show_control_page():
    """Live Gesture Control with persistent model loading"""
    st.header("🎮 Live Gesture Control")
    
    # Initialize model loading if not done
    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = False
    
    # Try to load model if not already loaded
    if not st.session_state.model_loaded:
        if load_saved_model():  # This function needs to be defined
            st.session_state.model_loaded = True
        else:
            st.error("""
            ❌ Model not loaded! Please:
            1. Train a model in the Train page
            2. Ensure model files exist in your directory
            """)
            if st.button("Go to Train Page"):
                st.session_state.page = "Train"
            return
    
    # Check mappings
    if not st.session_state.mappings:
        st.error("❌ No actions mapped! Please map gestures to actions.")
        if st.button("Go to Mapping Page"):
            st.session_state.page = "Map"
        return
    
    # Display current mappings
    with st.expander("📋 Current Mappings", expanded=True):
        for gesture, action in st.session_state.mappings.items():
            st.write(f"👉 {gesture.ljust(15)} → {action}")
    
    # Model information
    st.success(f"✅ Model Loaded: {MODEL_OPTIONS.get(st.session_state.model_type, 'Unknown')}")
    
    # Status indicators
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Gestures Trained", len(st.session_state.gesture_data))
    with col2:
        st.metric("Actions Mapped", len(st.session_state.mappings))
    with col3:
        st.metric("Model Ready", "Yes" if st.session_state.model else "No")
    
    # Main control interface
    frame_placeholder = st.empty()
    status_display = st.empty()
    stop_button = st.button("🛑 Stop Control")
    
    cap = cv2.VideoCapture(0)
    gesture_names = list(st.session_state.gesture_data.keys())
    
    while cap.isOpened() and not stop_button:
        ret, frame = cap.read()
        if not ret:
            continue
            
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = extract_landmarks(hand_landmarks)
                if landmarks is not None:
                    gesture_idx, confidence = predict_gesture(
                        st.session_state.model,
                        st.session_state.model_type,
                        landmarks
                    )
                    
                    # Draw landmarks
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )
                    
                    if gesture_idx is not None and confidence > 0.85:
                        gesture_name = gesture_names[gesture_idx]
                        
                        # Visual feedback
                        cv2.putText(frame, f"{gesture_name} ({confidence:.2f})", 
                                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        
                        if gesture_name in st.session_state.mappings:
                            action_name = st.session_state.mappings[gesture_name]
                            execute_action(action_name)
                            cv2.putText(frame, f"Action: {action_name}", 
                                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                            status_display.success(f"✅ Executing: {action_name.replace('_', ' ').title()}")
                        else:
                            status_display.warning(f"⚠️ Detected '{gesture_name}' but no action mapped")
        
        frame_placeholder.image(frame, channels="BGR", use_column_width=True)
        time.sleep(0.05)
    
    cap.release()
    if stop_button:
        status_display.success("🛑 Control stopped")

def load_saved_model():
    """Load model from file if exists"""
    model_types = ["SVM", "CNN", "LSTM"]
    for model_type in model_types:
        model_path = f"gesture_model_{model_type}.h5" if model_type != "SVM" else f"gesture_model_{model_type}.pkl"
        if Path(model_path).exists():
            try:
                if model_type == "SVM":
                    model = joblib.load(model_path)
                else:
                    model = load_model(model_path)
                st.session_state.model = model
                st.session_state.model_type = model_type
                return True
            except Exception as e:
                st.error(f"Error loading model: {str(e)}")
    return False


def main():
    """Main application entry point"""
    st.set_page_config(
        page_title="Gesture Control Pro",
        page_icon="👋",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Load CSS and set theme
    load_css()
    
    # Load saved data if not already loaded
    if not st.session_state.gesture_data:
        st.session_state.gesture_data, st.session_state.mappings = load_saved_data()
    
    # Enhanced sidebar with theme toggle
    with st.sidebar:
        st.markdown("<h1 style='text-align: center;'>Gesture Control Pro</h1>", unsafe_allow_html=True)
        
        # Theme toggle
        col1, col2 = st.columns([1, 2])
        with col1:
            if st.button("🌙" if st.session_state.dark_mode else "☀️"):
                st.session_state.dark_mode = not st.session_state.dark_mode
                st.rerun()
        with col2:
            st.markdown(f"**{'Dark' if st.session_state.dark_mode else 'Light'} Mode**")
        
        # Navigation
        st.markdown("---")
        menu_options = {
            "🏠 Home": "Home",
            "📸 Record Gesture": "Record",
            "🤖 Train Model": "Train",
            "📊 Compare Models": "Compare",
            "🔄 Map Gestures": "Map",
            "🎮 Live Control": "Control"
        }
        
        for icon, page in menu_options.items():
            if st.button(icon, use_container_width=True):
                st.session_state.page = page
        
        st.markdown("---")
        
        # Model status
        if st.session_state.model:
            st.success(f"Model: {MODEL_OPTIONS.get(st.session_state.model_type, 'Unknown')}")
        else:
            st.warning("No model loaded")
        
        # Quick actions
        if st.button("📚 Documentation"):
            webbrowser.open_new_tab("https://example.com/docs")
        if st.button("🐛 Report Issue"):
            webbrowser.open_new_tab("https://example.com/issues")
    
    # Page routing
    if st.session_state.page == "Home":
        show_home_page()
    elif st.session_state.page == "Record":
        show_record_page()
    elif st.session_state.page == "Train":
        show_train_page()
    elif st.session_state.page == "Compare":
        show_comparison_page()       
    elif st.session_state.page == "Map":
        show_mapping_page()
    elif st.session_state.page == "Control":
        show_control_page()

if __name__ == "__main__":

    main()
