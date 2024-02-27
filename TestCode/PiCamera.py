import time
import picamera

def capture_video(output_path="video.h264", duration=10):
    with picamera.PiCamera() as camera:
        camera.resolution = (1920, 1080)

        # Capture a video
        camera.start_recording(output_path)
        camera.wait_recording(duration)
        camera.stop_recording()

if __name__ == "__main__":
    output_filename = "video.h264"
    video_duration = 10  # in seconds
    capture_video(output_filename, video_duration)
    print(f"Video captured and saved as {output_filename}")
