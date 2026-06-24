#include "telemetry_hal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>

#define THERMAL_ZONE_PATH "/sys/class/thermal/thermal_zone0/temp"
#define PROC_STAT_PATH    "/proc/stat"
#define PROC_MEM_PATH     "/proc/meminfo"

static FILE *fd_temp = NULL;
static FILE *fd_stat = NULL;
static FILE *fd_mem  = NULL;

static uint64_t prev_idle = 0;
static uint64_t prev_total = 0;

int TelemetryHAL_Init(void) {
    fd_temp = fopen(THERMAL_ZONE_PATH, "r");
    fd_stat = fopen(PROC_STAT_PATH, "r");
    fd_mem  = fopen(PROC_MEM_PATH, "r");

    if (!fd_temp || !fd_stat || !fd_mem) {
        TelemetryHAL_Deinit();
        return -1; 
    }
    return 0; 
}

void TelemetryHAL_Deinit(void) {
    if (fd_temp) fclose(fd_temp);
    if (fd_stat) fclose(fd_stat);
    if (fd_mem)  fclose(fd_mem);
}

int TelemetryHAL_Update(TelemetryData_t *data) {
    if (!fd_temp || !fd_stat || !fd_mem || !data) return -1;

    // 1. Read Temperature (in millidegrees)
    rewind(fd_temp);
    int raw_temp;
    if (fscanf(fd_temp, "%d", &raw_temp) == 1) {
        data->cpu_temperature_c = raw_temp / 1000.0f; 
    }

    // 2. Read CPU Usage (Calculate Delta)
    rewind(fd_stat);
    char buffer[256];
    if (fgets(buffer, sizeof(buffer), fd_stat)) {
        uint64_t user, nice, system, idle, iowait, irq, softirq, steal;
        sscanf(buffer, "cpu %llu %llu %llu %llu %llu %llu %llu %llu",
               &user, &nice, &system, &idle, &iowait, &irq, &softirq, &steal);

        uint64_t current_idle = idle + iowait;
        uint64_t current_non_idle = user + nice + system + irq + softirq + steal;
        uint64_t current_total = current_idle + current_non_idle;

        uint64_t total_delta = current_total - prev_total;
        uint64_t idle_delta = current_idle - prev_idle;

        if (total_delta > 0) {
            data->cpu_usage_percent = (float)(total_delta - idle_delta) / total_delta * 100.0f;
        } else {
            data->cpu_usage_percent = 0.0f;
        }

        prev_idle = current_idle;
        prev_total = current_total;
    }

    // 3. Read Memory Usage
    rewind(fd_mem);
    uint32_t mem_total = 0, mem_free = 0, buffers = 0, cached = 0;
    
    while (fgets(buffer, sizeof(buffer), fd_mem)) {
        if (strncmp(buffer, "MemTotal:", 9) == 0) sscanf(buffer, "MemTotal: %u kB", &mem_total);
        else if (strncmp(buffer, "MemFree:", 8) == 0) sscanf(buffer, "MemFree: %u kB", &mem_free);
        else if (strncmp(buffer, "Buffers:", 8) == 0) sscanf(buffer, "Buffers: %u kB", &buffers);
        else if (strncmp(buffer, "Cached:", 7) == 0) {
            sscanf(buffer, "Cached: %u kB", &cached);
            break; // Stop parsing early to save cycles
        }
    }
    
    data->ram_total_mb = mem_total / 1024;
    data->ram_free_mb = (mem_free + buffers + cached) / 1024;

    return 0;
}