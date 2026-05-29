import time
import cv2

class RealTimeSignTransformer(VideoTransformerBase):
    def __init__(self, result_queue):
        self.result_queue = result_queue
        # ... [Your MediaPipe initialization code remains here] ...
        
        # Telemetry State Trackers
        self.prev_frame_time = 0
        self.new_frame_time = 0

    def transform(self, frame):
        # 1. Spatial Telemetry (Frame Width, Height, and Channels)
        img = frame.to_ndarray(format="bgr24")
        height, width, channels = img.shape
        
        # 2. Performance Telemetry (Calculating Actual Processing FPS)
        self.new_frame_time = time.time()
        fps = 1 / (self.new_frame_time - self.prev_frame_time)
        self.prev_frame_time = self.new_frame_time
        
        # Mirror image for natural movement
        img = cv2.flip(img, 1) 
        
        # Run your MediaPipe Tracking Matrix
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img)
        detection_result = self.landmarker.detect(mp_image)
        
        # 3. Positional Telemetry (Hand Coordinate Tracing)
        telemetry_log = "No Hands Tracked"
        
        if detection_result.hand_landmarks:
            for hand_landmarks in detection_result.hand_landmarks:
                # Extract specific point telemetry (e.g., Wrist location)
                wrist = hand_landmarks[0]
                
                # Convert normalized tracking decimals to actual pixel coordinates
                pixel_x = int(wrist.x * width)
                pixel_y = int(wrist.y * height)
                
                telemetry_log = f"Wrist Vector: X={pixel_x}, Y={pixel_y}"
                
                # Draw telemetry landmarks on frame
                for landmark in hand_landmarks:
                    x = int(landmark.x * width)
                    y = int(landmark.y * height)
                    cv2.circle(img, (x, y), 4, (230, 30, 150), -1)

        # --- RENDER TELEMETRY OVERLAY ON CAMERA SCREEN ---
        # Draw live stats directly onto the top corner of the camera feed
        cv2.putText(img, f"FPS: {int(fps)} | Res: {width}x{height}", (20, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(img, telemetry_log, (20, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return img
