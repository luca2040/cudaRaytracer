#pragma once

#include <SDL2/SDL.h>
#include <GL/glew.h>
#include "DrawValues.h"

void keyPressed(SDL_Keycode key);
void mouseMoved(int2_L mouse, int2_L pMouse);
void onSceneComposition();
void onSetupFrame(GLuint glTex);
void onClose();
void drawFrame(GLuint tex, GLuint pbo);