#ifndef TELEMETRY_HAL_H
#define TELEMETRY_HAL_H

#include <stdint.h>

typedef struct {
    float cpu_temperature_c;
    float cpu_usage_percent;
    uint32_t ram_total_mb;
    uint32_t ram_free_mb;
} TelemetryData_t;

int TelemetryHAL_Init(void);
void TelemetryHAL_Deinit(void);
int TelemetryHAL_Update(TelemetryData_t *data);

#endif // TELEMETRY_HAL_H