import cv2
import mediapipe as mp
import numpy as np
from datetime import datetime
import streamlit as st
from PIL import Image
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import joblib
import colorsys
import random

# Initialize MediaPipe
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Custom drawing specifications with purple color for mesh
purple_mesh_spec = mp_drawing.DrawingSpec(color=(250, 17, 219), thickness=1, circle_radius=1)

def generate_clothing_colors(skin_rgb, season, undertone):
    # Step 1: Normalize and convert RGB to HSL
    r, g, b = [x / 255.0 for x in skin_rgb]
    h, l, s = colorsys.rgb_to_hls(r, g, b)  # Hue, Lightness, Saturation

    hue_deg = h * 360
    analogous = [(hue_deg + 30) % 360, (hue_deg - 30) % 360]
    complementary = [(hue_deg + 150) % 360, (hue_deg + 180) % 360, (hue_deg + 210) % 360]
    triadic = [(hue_deg + 120) % 360, (hue_deg + 240) % 360]

    all_hues = analogous + complementary + triadic

    options = {
        "summer": (0.75, 0.55 * (1 - s)),
        "autumn": (0.35, 0.95 * (1 - s)),
        "winter": (0.40, 0.90 * (1 - s)),
        "spring": (0.50, 0.85 * (1 - s))
    }
    undertones = {
        "warm": np.array([45, 0, 0]),
        "neutral": np.array([0, 20, 0]),
        "cool": np.array([0, 0, 45])
    }
    transformation = np.array([45 * (1 - l), 0, 45 * l])
    
    results = [colorsys.hls_to_rgb(hue/360, options[season.lower()][0], options[season.lower()][1]) for hue in all_hues]
    return np.random.permutation((np.array(results) * 255 + undertones[undertone.lower()] + transformation) % 360)

def main():
    st.set_page_config(page_title="koreai", layout="wide")
    
    # Initialize session state
    if 'current_screen' not in st.session_state:
        st.session_state.current_screen = "welcome"

    # Clear the entire UI before showing any screen
    st.empty()

    if 'feature_colors' not in st.session_state:
        st.session_state.feature_colors = {}
    if 'recording' not in st.session_state:
        st.session_state.recording = False
    if 'show_mesh' not in st.session_state:
        st.session_state.show_mesh = True
    if 'captured_frame' not in st.session_state:
        st.session_state.captured_frame = None
    if st.session_state.current_screen == "explore":
        show_explore_screen()
    
    # Screen navigation
    if st.session_state.current_screen == "welcome":
        show_welcome_screen()
    elif st.session_state.current_screen == "camera":
        show_camera_screen()
    elif st.session_state.current_screen == "results":
        show_results_screen()

def show_welcome_screen():
    st.title("Welcome to koreai!")
    st.markdown("""
    Discover your perfect color palette based on your natural features!
    
    This analysis will:
    - Capture your eye, lip, cheek, and hair colors
    - Determine your undertone
    - Identify your seasonal color palette
    - Show you which colors complement you best
                
    On the next screen, position your face in the frame. Do not blink or move your face for 5 seconds.
    """)
    
    if st.button("Begin analysis", key="start_btn"):
        st.session_state.current_screen = "camera"
        st.rerun()

def show_camera_screen():
    st.title("Color capture")
    st.markdown("""
    Position your face in the frame. Do not blink or move your face for 5 seconds.
    """)
    
    # Place the stop button outside the camera loop
    if st.button("Stop early", key="stop_btn_unique"):
        if 'cap' in st.session_state:
            st.session_state.cap.release()
            del st.session_state.cap
        st.session_state.current_screen = "results"
        st.rerun()
    
    # Countdown and status
    countdown_placeholder = st.empty()
    status_placeholder = st.empty()
    
    # Camera feed placeholder
    camera_placeholder = st.empty()
    
    # Initialize camera and store in session state
    if 'cap' not in st.session_state:
        st.session_state.cap = cv2.VideoCapture(0)
        st.session_state.start_time = datetime.now()
    
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:
        
        while st.session_state.current_screen == "camera":
            current_time = datetime.now()
            elapsed = (current_time - st.session_state.start_time).total_seconds()
            remaining = max(0, 5 - elapsed)
            
            # Update countdown
            countdown_placeholder.markdown(f"### Time remaining: {int(remaining)} seconds")
            
            ret, frame = st.session_state.cap.read()
            if not ret:
                st.error("Failed to capture frame from camera")
                break
            
            # Convert to RGB for MediaPipe
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(image_rgb)
            
            # Make a copy for visualization
            display_frame = frame.copy()
            
            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                
                # Always show mesh during analysis
                mp_drawing.draw_landmarks(
                    image=display_frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=purple_mesh_spec
                )
                
                # Define feature points
                features = {
                    'Eye': 468,   # Left iris center
                    'Lips': 13,    # Lower lip center
                    'Cheek': 205,  # Left cheek
                    'Hair': 295    # Eyebrow, in place for hair color
                }
                
                # Reset feature colors for this frame
                st.session_state.feature_colors = {}
                
                for feature, idx in features.items():
                    color_bgr, (x, y) = sample_feature_color(frame, face_landmarks, idx)
                    if color_bgr is not None:
                        st.session_state.feature_colors[feature] = color_bgr
                        
                        # Draw feature point and swatch
                        cv2.circle(display_frame, (x, y), 3, (255, 255, 255), -1)
                        swatch_x = x + 10
                        swatch_y = y - 10
                        r, g, b = int(color_bgr[2]), int(color_bgr[1]), int(color_bgr[0])
                        cv2.rectangle(display_frame, (swatch_x, swatch_y), 
                                     (swatch_x + 40, swatch_y + 20), (b, g, r), -1)
                        cv2.rectangle(display_frame, (swatch_x, swatch_y), 
                                     (swatch_x + 40, swatch_y + 20), (0, 0, 0), 1)
            
            # Convert to RGB for Streamlit
            display_frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            camera_placeholder.image(display_frame_rgb, channels="RGB", width=800)
            
            # Check if analysis time is complete
            if elapsed >= 5:
                status_placeholder.success("Analysis complete!")
                st.session_state.cap.release()
                del st.session_state.cap
                st.session_state.captured_frame = display_frame_rgb
                st.session_state.current_screen = "results"
                st.rerun()
                break
    
    # If we exit the loop without completing
    if st.session_state.current_screen == "camera":
        if 'cap' in st.session_state:
            st.session_state.cap.release()
            del st.session_state.cap
        st.error("Analysis interrupted")
        if st.button("Try again", key="try_again_btn"):
            st.session_state.current_screen = "camera"
            st.rerun()
        if st.button("Back to welcome", key="back_welcome_btn"):
            st.session_state.current_screen = "welcome"
            st.rerun()

def show_results_screen():
    st.markdown('<div class="fade-in">', unsafe_allow_html=True)
    st.title("Your color analysis")

    if not st.session_state.feature_colors:
        st.warning("No color data captured. Please go back and try again.")
        if st.button("Back to camera"):
            st.session_state.current_screen = "camera"
            st.rerun()
        return

    features_df = prepare_feature_data(st.session_state.feature_colors)
    undertone = predict_undertone(features_df)
    season = predict_season(features_df)

    st.subheader("Your features")
    if st.session_state.captured_frame is not None:
        st.image(st.session_state.captured_frame, channels="RGB", width=800)

    if st.button("Analyze again", key="analyze_again"):
        st.session_state.current_screen = "camera"
        st.rerun()

    st.subheader("Captured colors")
    for feature, color in st.session_state.feature_colors.items():
        r, g, b = int(color[2]), int(color[1]), int(color[0])
        hex_color = "#{:02x}{:02x}{:02x}".format(r, g, b).upper()
        st.markdown(f'<div style="width: 30px; height: 30px; background-color: {hex_color}; border: 1px solid black;"></div>', 
                   unsafe_allow_html=True)
        st.write(f"{feature}: RGB({r}, {g}, {b}) {hex_color}")

    st.subheader("Your analysis")
    st.markdown(f"### Undertone: {undertone[0].capitalize()}")
    if undertone[0] == "Cool":
        st.markdown("""
        - Your skin has pink, red, or blueish hues
        - Silver jewelry typically looks better on you
        - You look best in jewel tones and cool pastels
        """)
    elif undertone[0] == "Warm":
        st.markdown("""
        - Your skin has yellow, peach, or golden hues
        - Gold jewelry typically looks better on you
        - You look best in earth tones and warm colors
        """)
    elif undertone[0] == "Neutral":
        st.markdown("""
        - Your skin has a balance of warm and cool tones
        - Both gold and silver jewelry work well
        - You can wear a wide range of colors
        """)

    st.markdown(f"### Season: {season[0].capitalize()}")
    display_seasonal_palette(season[0])
    
    # Add custom color generation based on cheek color (skin tone)
    st.markdown("### Your personalized color palette")
    cheek_rgb = [int(st.session_state.feature_colors['Cheek'][2]), 
                int(st.session_state.feature_colors['Cheek'][1]), 
                int(st.session_state.feature_colors['Cheek'][0])]
    custom_colors = generate_clothing_colors(cheek_rgb, season[0], undertone[0])
    
    # Display custom colors
    cols = st.columns(len(custom_colors))
    for i, color in enumerate(custom_colors):
        r, g, b = int(color[0]), int(color[1]), int(color[2])
        hex_color = "#{:02x}{:02x}{:02x}".format(r, g, b).upper()
        with cols[i]:
            st.markdown(f'<div style="width: 40px; height: 40px; background-color: {hex_color}; border: 1px solid black;"></div>', 
                      unsafe_allow_html=True)
            st.caption(hex_color)
    
    st.markdown("""
    These colors are uniquely generated based on your skin tone and season.
    They're designed to complement your natural coloring perfectly.
    """)

    if st.button("Explore more", key="explore_more"):
        st.session_state.current_screen = "explore"
        st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

def show_explore_screen():
    # Clear the entire UI before showing any screen
    st.empty()

    st.title("Explore your palette in real time")
    st.markdown("Click the 'Next color' button to view your custom colors against you, like a Korean color analysis!")
    
    if 'cap' not in st.session_state:
        st.session_state.cap = cv2.VideoCapture(0)
    
    # Determine season based on features    
    season = predict_season(prepare_feature_data(st.session_state.feature_colors))[0]

    # Determine undertone based on features    
    undertone = predict_undertone(prepare_feature_data(st.session_state.feature_colors))[0]
    
    # Get custom colors based on cheek color (skin tone)
    cheek_rgb = [int(st.session_state.feature_colors['Cheek'][2]), 
                int(st.session_state.feature_colors['Cheek'][1]), 
                int(st.session_state.feature_colors['Cheek'][0])]
    custom_colors = generate_clothing_colors(cheek_rgb, season, undertone)
    
    # Add standard palette colors as well for comparison
    standard_colors = display_palette_for_overlay(season)
    
    # Combine both sets of colors
    all_colors = list(custom_colors) + standard_colors
    
    # Initialize the current color index if it doesn't exist
    if 'current_color_index' not in st.session_state:
        st.session_state.current_color_index = 0
    
    # Add option to switch between standard and custom colors
    color_mode = st.radio(
        "Color palette:",
        ["Custom colors (based on your skin)", "Standard seasonal colors"],
        horizontal=True
    )
    
    # Use appropriate color set based on selection
    if color_mode == "Custom colors (based on your skin)":
        colors = custom_colors
        st.info("These colors are generated uniquely for your skin tone")
    else:
        colors = standard_colors
        st.info("These are the standard colors for your season")

    # Display undertone information
    st.markdown(f"Your undertone: <strong>{undertone.capitalize()}</strong>", unsafe_allow_html=True)
    
    # Display season information
    st.markdown(f"Your color season: <strong>{season.capitalize()}</strong>", unsafe_allow_html=True)

    # Display current color name and hex
    color_hex = "#{:02x}{:02x}{:02x}".format(
        int(colors[st.session_state.current_color_index][0]),
        int(colors[st.session_state.current_color_index][1]),
        int(colors[st.session_state.current_color_index][2])
    )
    st.markdown(f"Color {st.session_state.current_color_index + 1}/{len(colors)}: <strong>{color_hex}</strong>", unsafe_allow_html=True)

    next_color = st.button("Next color", key="next_color")
    if next_color:
        st.session_state.current_color_index = (st.session_state.current_color_index + 1) % len(colors)
    
    # Create camera placeholder
    camera_placeholder = st.empty()

    # Back button at the top
    stop = st.button("Back to results", key="back_to_results")
    if stop:
        if 'cap' in st.session_state:
            st.session_state.cap.release()
            del st.session_state.cap
        st.session_state.current_screen = "results"
        st.rerun()
        
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:
        while not stop:
            ret, frame = st.session_state.cap.read()
            if not ret:
                st.error("Failed to capture frame from camera")
                break
            
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(image_rgb)
            display_frame = frame.copy()
            
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0]
                h, w, _ = frame.shape
                
                # Get chin landmark position
                chin_x = int(landmarks.landmark[152].x * w)
                chin_y = int(landmarks.landmark[152].y * h)
                
                # Get face width for better rectangle sizing
                left_face = int(landmarks.landmark[234].x * w)
                right_face = int(landmarks.landmark[454].x * w)
                face_width = right_face - left_face
                
                # Create a large rectangle under the chin
                # Korean style typically uses a large rectangle
                rect_width = int(face_width * 3)  # Make wider than face
                rect_height = int(rect_width * 1)  # Proportional height
                
                # Position the rectangle centered under the chin
                rect_x = chin_x - (rect_width // 2)
                rect_y = chin_y + 30  # Some space below the chin
                
                # Get current color
                current_color = colors[st.session_state.current_color_index]
                r, g, b = int(current_color[0]), int(current_color[1]), int(current_color[2])
                
                # Draw the large color rectangle
                cv2.rectangle(display_frame, 
                              (rect_x, rect_y), 
                              (rect_x + rect_width, rect_y + rect_height), 
                              (b, g, r), -1)  # Filled rectangle
                
                # Add border for better visibility
                cv2.rectangle(display_frame, 
                              (rect_x, rect_y), 
                              (rect_x + rect_width, rect_y + rect_height), 
                              (0, 0, 0), 2)  # Black border
            
            # Display the frame with fixed width
            frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            camera_placeholder.image(frame_rgb, channels="RGB", width=800)
    
    # Ensure camera is released when the function exits
    if 'cap' in st.session_state:
        st.session_state.cap.release()
        del st.session_state.cap

    st.container()

def display_palette_for_overlay(season):
    # Korean color analysis typically uses more refined palettes
    palette_hex = {
        "winter": ["#8b1f60", "#0a6c70", "#c4e9fc", "#c1c8e4", "#ed819b", "#742872"],
        "summer": ["#A7C7E7", "#C8A2C8", "#E6E6FA", "#FFB6C1", "#98FF98", "#F0FFF0"],
        "autumn": ["#2a635f", "#d08465", "#9d222d", "#f5ece4", "#bf6766", "#6c724c"],
        "spring": ["#FFD700", "#FFA07A", "#98FB98", "#87CEFA", "#FF69B4", "#FFFACD"]
    }
    
    hex_to_rgb = lambda hx: tuple(int(hx.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
    return [hex_to_rgb(color) for color in palette_hex[season]]

def display_seasonal_palette(season):
    palettes = {
        "winter": ["#8b1f60", "#0a6c70", "#c4e9fc", "#c1c8e4", "#ed819b", "#742872"],
        "summer": ["#A7C7E7", "#C8A2C8", "#E6E6FA", "#FFB6C1", "#98FF98", "#F0FFF0"],
        "autumn": ["#2a635f", "#d08465", "#9d222d", "#f5ece4", "#bf6766", "#6c724c"],
        "spring": ["#FFD700", "#FFA07A", "#98FB98", "#87CEFA", "#FF69B4", "#FFFACD"]
    }

    st.markdown("Your recommended colors")
    cols = st.columns(len(palettes[season]))
    for i, color in enumerate(palettes[season]):
        with cols[i]:
            st.markdown(f'<div style="width: 40px; height: 40px; background-color: {color}; border: 1px solid black;"></div>', 
                       unsafe_allow_html=True)
            st.caption(color)

    st.markdown("""
    Tips for your season
    - These colors will complement your natural coloring
    - Try incorporating them into your wardrobe and makeup
    - Use the darker shades for formal wear and lighter shades for casual
    """)

# Helper functions from original code
def sample_feature_color(frame, landmarks, idx, region_size=6):
    h, w, _ = frame.shape
    x = int(landmarks.landmark[idx].x * w)
    y = int(landmarks.landmark[idx].y * h)
    patch = frame[y-region_size:y+region_size, x-region_size:x+region_size]
    if patch.size == 0:
        return None, (x, y)
    avg_bgr = np.average(np.average(patch, axis=0), axis=0)
    return avg_bgr, (x, y)

def prepare_feature_data(feature_colors):
    data = {
        'Eye_R': [int(feature_colors['Eye'][2])],
        'Eye_G': [int(feature_colors['Eye'][1])],
        'Eye_B': [int(feature_colors['Eye'][0])],
        'Lips_R': [int(feature_colors['Lips'][2])],
        'Lips_G': [int(feature_colors['Lips'][1])],
        'Lips_B': [int(feature_colors['Lips'][0])],
        'Cheek_R': [int(feature_colors['Cheek'][2])],
        'Cheek_G': [int(feature_colors['Cheek'][1])],
        'Cheek_B': [int(feature_colors['Cheek'][0])],
        'Hair_R': [int(feature_colors['Hair'][2])],
        'Hair_G': [int(feature_colors['Hair'][1])],
        'Hair_B': [int(feature_colors['Hair'][0])]
    }
    return pd.DataFrame(data)

undertone_pipeline = joblib.load('models/undertone_classifier.pkl')
season_pipeline = joblib.load('models/season_classifier.pkl')

def predict_undertone(features_df):
    prediction = undertone_pipeline.predict(features_df)
    return prediction.tolist() 

def predict_season(features_df):
    prediction = season_pipeline.predict(features_df)
    return [prediction[0].lower()]

if __name__ == "__main__":
    main()