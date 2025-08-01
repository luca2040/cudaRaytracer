#include "Scene.h"

Scene *scene = nullptr; // CPU scene

Scene *d_scene; // GPU scene copy
size_t sceneStructSize = sizeof(Scene);