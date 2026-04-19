import psutil
import time

while True:
    temps = psutil.sensors_temperatures()
    
    if not temps:
        print("No temperature sensors found (may need admin rights)")
    else:
        for chip, entries in temps.items():
            for entry in entries:
                print(f"[{chip}] {entry.label or 'CPU'}: {entry.current:.1f}°C "
                      f"(high={entry.high}, crit={entry.critical})")
    
    print("---")
    time.sleep(1)