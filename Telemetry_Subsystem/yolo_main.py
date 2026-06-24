import ctypes
import time
import os
import cv2
from ultralytics import YOLO

# 1. Map the C struct accurately into Python memory
class TelemetryData(ctypes.Structure):
    _fields_ = [
        ("cpu_temperature_c", ctypes.c_float),
        ("cpu_usage_percent", ctypes.c_float),
        ("ram_total_mb", ctypes.c_uint32),
        ("ram_free_mb", ctypes.c_uint32)
    ]

# 2. Load the optimized Shared Library
lib_path = os.path.abspath('./libtelemetry.so')
if not os.path.exists(lib_path):
    raise FileNotFoundError("libtelemetry.so not found. Run 'make' first.")

hal = ctypes.CDLL(lib_path)

# 3. Define C function signatures
hal.TelemetryHAL_Init.restype = ctypes.c_int
hal.TelemetryHAL_Update.argtypes = [ctypes.POINTER(TelemetryData)]
hal.TelemetryHAL_Update.restype = ctypes.c_int
hal.TelemetryHAL_Deinit.restype = None

def main():
    # Initialize the C HAL once
    if hal.TelemetryHAL_Init() != 0:
        print("Error: Failed to initialize Telemetry HAL.")
        return

    data = TelemetryData()

    print("Loading YOLOv8 Nano...")
    # Using the Nano model is highly recommended for the Pi 5's CPU
    model = YOLO("yolov8n.pt") 

    # Initialize video capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Camera not found.")
        hal.TelemetryHAL_Deinit()
        return

    print("Starting Main Loop... Press 'q' to quit.")
    
    try:
        while True:
            # Step A: Capture Video Frame
            ret, frame = cap.read()
            if not ret:
                break

            # Step B: Pull Hardware Telemetry instantly from C space
            hal.TelemetryHAL_Update(ctypes.byref(data))
            
            # Step C: YOLO Inference
            # verbose=False suppresses ultralytics console spam so we can see telemetry
            results = model(frame, verbose=False)

            # Step D: Print Telemetry Log
            print(f"[HW STATE] Temp: {data.cpu_temperature_c:.1f}°C | "
                  f"CPU: {data.cpu_usage_percent:.1f}% | "
                  f"RAM Free: {data.ram_free_mb}/{data.ram_total_mb} MB")

            # Optional: Display the frame with bounding boxes
            # If you are running completely headless (no GUI), comment out these 3 lines
            annotated_frame = results[0].plot()
            cv2.imshow("YOLO Inference", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\nExiting cleanly...")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        hal.TelemetryHAL_Deinit()

if __name__ == "__main__":
    main()