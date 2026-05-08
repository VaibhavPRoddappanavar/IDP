import threading
import time
import psutil
import platform

class SystemMonitor:
    """Non-blocking hardware monitor for CPU-only metrics."""

    def __init__(self, poll_interval: float = 0.5):
        self.poll_interval = poll_interval
        self._running = False
        self._thread = None

        self._cpu_usage = 0.0
        self._ram_usage = 0.0
        self._temperature = 0.0

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def snapshot(self) -> dict:
        return {
            "cpu_usage_percent": self._cpu_usage,
            "ram_usage_percent": self._ram_usage,
            "temperature": self._temperature,
        }

    def _monitor_loop(self):
        while self._running:
            self._cpu_usage = psutil.cpu_percent(interval=None)
            self._ram_usage = psutil.virtual_memory().percent
            self._temperature = self._get_cpu_temperature()
            time.sleep(self.poll_interval)

    def _get_cpu_temperature(self) -> float:
        """Fetch CPU temperature depending on the OS."""
        sys_os = platform.system()
        temp = 0.0
        try:
            if sys_os == "Linux":
                # Raspberry Pi specific temp file
                with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
                    temp = float(f.read().strip()) / 1000.0
            elif sys_os == "Windows":
                import wmi
                w = wmi.WMI(namespace="root\\wmi")
                temp_info = w.MSAcpi_ThermalZoneTemperature()
                if temp_info:
                    temp = (temp_info[0].CurrentTemperature / 10.0) - 273.15
        except Exception:
            temp = 0.0
        return temp
