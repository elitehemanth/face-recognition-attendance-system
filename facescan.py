import streamlit as st
import cv2
import pandas as pd
import json
from datetime import datetime
from deepface import DeepFace
import os
import numpy as np

# Paths for database and logs
DB_PATH = "faces_db"
os.makedirs(DB_PATH, exist_ok=True)
JSON_FILE = "attendance.json"

def load_attendance_data():
    """Load attendance data from JSON file"""
    try:
        if os.path.exists(JSON_FILE):
            with open(JSON_FILE, 'r') as file:
                data = json.load(file)
                return pd.DataFrame(data)
        else:
            return pd.DataFrame(columns=["Name", "Type", "Time"])
    except (json.JSONDecodeError, FileNotFoundError):
        return pd.DataFrame(columns=["Name", "Type", "Time"])

def save_attendance_data(df):
    """Save attendance data to JSON file"""
    try:
        data = df.to_dict('records')
        with open(JSON_FILE, 'w') as file:
            json.dump(data, file, indent=4)
        return True
    except Exception as e:
        st.error(f"Error saving JSON: {e}")
        return False

def log_check(name, check_type):
    """Log attendance check-in/check-out"""
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Load existing data
    df = load_attendance_data()
    
    # Add new entry
    new_entry = pd.DataFrame([{"Name": name, "Type": check_type, "Time": now}])
    df = pd.concat([df, new_entry], ignore_index=True)
    
    # Save updated data
    if save_attendance_data(df):
        st.success(f"✅ {check_type} successfully recorded for {name} at {now}")
        st.balloons()  # Celebration animation
        return True
    else:
        st.error(f"❌ Failed to record {check_type} for {name}")
        return False

@st.cache_resource
def get_webcam():
    """Get webcam with caching to avoid multiple instances"""
    return cv2.VideoCapture(0)

def recognize_face(frame):
    """Recognize face using DeepFace"""
    if not os.path.exists(DB_PATH) or len(os.listdir(DB_PATH)) == 0:
        return None
        
    temp_img = "temp_live.jpg"
    cv2.imwrite(temp_img, frame)
    
    files = [f for f in os.listdir(DB_PATH) if f.endswith('.jpg')]
    recognized = None
    
    try:
        for file in files:
            db_img_path = os.path.join(DB_PATH, file)
            try:
                # Use DeepFace.verify with proper parameters
                result = DeepFace.verify(
                    img1_path=temp_img, 
                    img2_path=db_img_path, 
                    enforce_detection=False,
                    model_name='VGG-Face'  # Specify model for consistency
                )
                
                if result["verified"]:
                    recognized = os.path.splitext(file)[0]
                    break
                    
            except Exception as e:
                st.warning(f"Recognition error with {file}: {str(e)}")
                continue
                
    except Exception as e:
        st.error(f"Overall recognition error: {str(e)}")
    
    finally:
        if os.path.exists(temp_img):
            os.remove(temp_img)
    
    return recognized

def add_face():
    """Register new face"""
    st.subheader("👤 Register New User")
    
    # Initialize session state for face registration
    if 'capture_face' not in st.session_state:
        st.session_state.capture_face = False
    if 'face_name' not in st.session_state:
        st.session_state.face_name = ""
    
    name = st.text_input("Enter new user name to register:", value=st.session_state.face_name)
    st.session_state.face_name = name
    
    if st.button("📸 Capture Face", type="primary"):
        if not name.strip():
            st.error("⚠️ Please enter a valid name.")
            return
            
        st.session_state.capture_face = True
    
    if st.session_state.capture_face:
        with st.spinner("Accessing webcam..."):
            cam = cv2.VideoCapture(0)
            if not cam.isOpened():
                st.error("❌ Failed to access webcam.")
                st.session_state.capture_face = False
                return
                
            ret, frame = cam.read()
            cam.release()
            
            if not ret:
                st.error("❌ Failed to capture image from webcam.")
                st.session_state.capture_face = False
                return
        
        # Save the captured image
        filename = os.path.join(DB_PATH, f"{name.strip()}.jpg")
        cv2.imwrite(filename, frame)
        
        # Display captured image
        st.image(frame, channels="BGR", caption=f"Registered image for {name.strip()}", width=300)
        st.success(f"✅ Face successfully registered for **{name.strip()}**!")
        st.info(f"📁 Image saved as: {filename}")
        
        # Reset the capture state
        st.session_state.capture_face = False
        st.session_state.face_name = ""

def webcam_verification(check_type):
    """Perform webcam-based face verification"""
    st.subheader(f"🎯 {check_type} Verification")
    
    # Check if any faces are registered
    if not os.path.exists(DB_PATH) or len([f for f in os.listdir(DB_PATH) if f.endswith('.jpg')]) == 0:
        st.warning("⚠️ No registered faces found! Please register a face first.")
        return
    
    # Initialize session state for this verification session
    session_key = f"{check_type.lower()}_verification"
    if session_key not in st.session_state:
        st.session_state[session_key] = {
            'frame': None,
            'verification_done': False,
            'last_capture_time': None
        }
    
    # Capture new frame button
    if st.button(f"📹 Capture Frame for {check_type}", key=f"capture_{check_type}"):
        with st.spinner("Capturing frame..."):
            cam = cv2.VideoCapture(0)
            if not cam.isOpened():
                st.error("❌ Webcam access problem. Please check your camera.")
                return
                
            ret, frame = cam.read()
            cam.release()
            
            if ret:
                st.session_state[session_key]['frame'] = frame
                st.session_state[session_key]['verification_done'] = False
                st.session_state[session_key]['last_capture_time'] = datetime.now()
                st.success("📸 Frame captured successfully!")
            else:
                st.error("❌ Failed to capture frame.")
                return
    
    # Display captured frame if available
    if st.session_state[session_key]['frame'] is not None:
        st.image(
            st.session_state[session_key]['frame'], 
            channels="BGR", 
            caption=f"Captured frame for {check_type}", 
            width=400
        )
        
        # Show capture time
        if st.session_state[session_key]['last_capture_time']:
            st.info(f"🕐 Frame captured at: {st.session_state[session_key]['last_capture_time'].strftime('%H:%M:%S')}")
        
        # Verification button
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button(f"🔍 Verify & {check_type}", key=f"verify_{check_type}", type="primary"):
                with st.spinner("🔍 Analyzing face..."):
                    try:
                        name = recognize_face(st.session_state[session_key]['frame'])
                        
                        if name:
                            # Success - Face recognized
                            st.success(f"🎉 **Face verified successfully!**")
                            st.success(f"👤 Welcome, **{name}**!")
                            
                            # Add success overlay to frame
                            success_frame = st.session_state[session_key]['frame'].copy()
                            cv2.putText(success_frame, f"{check_type} - {name}", (20, 40),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            cv2.putText(success_frame, "VERIFIED", (20, 80),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            
                            st.image(success_frame, channels="BGR", caption=f"✅ {check_type} verified for {name}", width=400)
                            
                            # Log the attendance
                            if log_check(name, check_type):
                                st.session_state[session_key]['verification_done'] = True
                        else:
                            # Failed recognition
                            st.error("❌ **Face not recognized!**")
                            st.warning("Please try the following:")
                            st.write("• Ensure good lighting")
                            st.write("• Face the camera directly")
                            st.write("• Remove glasses/masks if worn during registration")
                            st.write("• Check if your face is registered")
                            
                            # Add failure overlay to frame
                            fail_frame = st.session_state[session_key]['frame'].copy()
                            cv2.putText(fail_frame, "NOT RECOGNIZED", (20, 40),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                            
                            st.image(fail_frame, channels="BGR", caption="❌ Face not recognized", width=400)
                            
                    except Exception as e:
                        st.error(f"❌ Verification failed: {str(e)}")
        
        with col2:
            if st.button(f"🔄 Recapture Frame", key=f"recapture_{check_type}"):
                # Clear the current frame and capture a new one
                st.session_state[session_key]['frame'] = None
                st.session_state[session_key]['verification_done'] = False
                st.rerun()
        
        # Reset button
        if st.session_state[session_key]['verification_done']:
            if st.button(f"🔄 Start New {check_type}", key=f"reset_{check_type}"):
                st.session_state[session_key] = {
                    'frame': None,
                    'verification_done': False,
                    'last_capture_time': None
                }
                st.rerun()
    
    else:
        st.info(f"👆 Click the button above to capture a frame for {check_type} verification")

def display_attendance_log():
    """Display attendance log from JSON file"""
    st.subheader("📊 Attendance Log")
    
    df = load_attendance_data()
    
    if not df.empty:
        # Display summary statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Records", len(df))
        with col2:
            unique_people = df['Name'].nunique() if 'Name' in df.columns else 0
            st.metric("Unique People", unique_people)
        with col3:
            today = datetime.now().strftime('%Y-%m-%d')
            today_records = len(df[df['Time'].str.startswith(today)]) if 'Time' in df.columns else 0
            st.metric("Today's Records", today_records)
        
        st.markdown("---")
        
        # Display the dataframe
        st.dataframe(
            df,
            use_container_width=True,
            column_config={
                "Name": st.column_config.TextColumn("👤 Name"),
                "Type": st.column_config.TextColumn("📋 Type"),
                "Time": st.column_config.DatetimeColumn("🕐 Timestamp"),
            }
        )
        
        # Option to download the data
        st.download_button(
            label="💾 Download Attendance Data (JSON)",
            data=json.dumps(df.to_dict('records'), indent=4),
            file_name=f"attendance_log_{datetime.now().strftime('%Y%m%d')}.json",
            mime="application/json"
        )
        
        # Display raw JSON (expandable)
        with st.expander("🔍 View Raw JSON Data"):
            st.json(df.to_dict('records'))
    else:
        st.info("📝 No attendance records found. Start by registering faces and logging attendance!")

# Main Streamlit App
st.set_page_config(
    page_title="Face Recognition Attendance System",
    page_icon="👤",
    layout="wide"
)

st.title("🎯 Face Recognition Attendance System")
st.markdown("**Powered by elitehemanth**")

# Create tabs
tab1, tab2 = st.tabs(["🎛️ Control Panel", "📊 Attendance Log"])

with tab1:
    # Face registration section
    add_face()
    
    st.markdown("---")
    
    # Check-in/Check-out section
    st.subheader("⏰ Check-In / Check-Out")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🟢 Start Check-In", type="primary", use_container_width=True):
            st.session_state.current_action = "Check-In"
    
    with col2:
        if st.button("🔴 Start Check-Out", type="secondary", use_container_width=True):
            st.session_state.current_action = "Check-Out"
    
    # Show verification interface if an action is selected
    if 'current_action' in st.session_state:
        st.markdown("---")
        webcam_verification(st.session_state.current_action)

with tab2:
    display_attendance_log()

# Sidebar with app info
with st.sidebar:
    st.header("ℹ️ App Information")
    st.write("**Face Database:**", DB_PATH)
    st.write("**Data Storage:**", JSON_FILE)
    
    # Display registered faces count
    if os.path.exists(DB_PATH):
        face_count = len([f for f in os.listdir(DB_PATH) if f.endswith('.jpg')])
        st.metric("Registered Faces", face_count)
    
    if face_count > 0:
        st.write("**Registered Users:**")
        for f in os.listdir(DB_PATH):
            if f.endswith('.jpg'):
                st.write(f"• {os.path.splitext(f)[0]}")
    
    st.markdown("---")
    st.markdown("**Features:**")
    st.markdown("- 🎯 Real-time face recognition")
    st.markdown("- 📊 JSON-based data storage")
    st.markdown("- 📱 Interactive web interface")
    st.markdown("- 💾 Data export functionality")
    
    # Clear all session state button
    if st.button("🔄 Reset All Sessions"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
