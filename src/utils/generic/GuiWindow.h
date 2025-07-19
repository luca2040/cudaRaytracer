#pragma once

#include <iostream>
#include <SDL2/SDL.h>
#include <GL/glew.h>

#include "../DrawValues.h"

#include "../../third_party/imgui/imgui.h"
// #include "../../third_party/imgui/imgui_impl_sdl2.h"
// #include "../../third_party/imgui/imgui_impl_opengl3.h"

class GuiWindow
{
private:
public:
  ImGuiIO *io = nullptr;
  const GLubyte *openGLversion;

  inline void RenderGui()
  {
    // Status window
    {
      ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_Once);
      ImGui::SetNextWindowSize(ImVec2(800, 300), ImGuiCond_Once);

      ImGui::Begin("Status");

      ImGui::Text("OpenGL version: %s", openGLversion);

      ImGui::Separator();

      ImGui::Text("FPS: %.1f", io->Framerate);
      ImGui::Text("DeltaTime: %.4f s", io->DeltaTime);

      ImGui::End();
    }

    // Controls window
    {
      constexpr const ImVec2 windowSize = ImVec2(800, 300);
      constexpr const ImVec2 windwoPos = ImVec2(WIDTH - windowSize.x - 10, 10);

      ImGui::SetNextWindowPos(windwoPos, ImGuiCond_Once);
      ImGui::SetNextWindowSize(windowSize, ImGuiCond_Once);

      ImGui::Begin("Controls");

      // Still to implement

      ImGui::End();
    }
  }

  GuiWindow() = default;
};

extern GuiWindow guiWindow;