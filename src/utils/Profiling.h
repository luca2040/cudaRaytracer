#pragma once

// Some colors to make colored profiler tags
#define PROFILER_RED 0x880000
#define PROFILER_GREEN 0x00FF00
#define PROFILER_BLUE 0x0000FF
#define PROFILER_YELLOW 0xFFFF00
#define PROFILER_CYAN 0x00FFFF
#define PROFILER_MAGENTA 0xFF00FF
#define PROFILER_ORANGE 0xDD9400
#define PROFILER_PURPLE 0x800080
#define PROFILER_PINK 0x99708B
#define PROFILER_BROWN 0x8B4513
#define PROFILER_GRAY 0x808080
#define PROFILER_DARK_GREEN 0x006400
#define PROFILER_LIME_GREEN 0x32CD32
#define PROFILER_LIGHT_BLUE 0xADD8E6
#define PROFILER_GOLD 0xFFD700
#define PROFILER_TURQUOISE 0x40E0D0

// #define ENABLE_PROFILING

#ifdef ENABLE_PROFILING

#include "../third_party/tracy/tracy/Tracy.hpp"
#include "../third_party/tracy/tracy/TracyC.h"

#define FRAMEMARK FrameMark
#define ZONESCOPED ZoneScoped
#define ZONESCOPEDN(name) ZoneScopedN(name)
#define ZONESCOPEDNC(name, color) ZoneScopedNC(name, color)
#define TRACYCZONEN(tag, name, active) TracyCZoneN(tag, name, active)
#define TRACYCZONENC(tag, name, active, color) \
  TracyCZoneN(tag, name, active);              \
  TracyCZoneColor(tag, color)
#define TRACYCZONEEND(tag) TracyCZoneEnd(tag)

#else

#define FRAMEMARK
#define ZONESCOPED
#define ZONESCOPEDN(name)
#define ZONESCOPEDNC(name, color)
#define TRACYCZONEN(tag, name, active)
#define TRACYCZONENC(tag, name, active, color)
#define TRACYCZONEEND(tag)

#endif