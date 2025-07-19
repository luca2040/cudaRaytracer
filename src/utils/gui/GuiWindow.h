#pragma once

#include <iostream>
#include <SDL2/SDL.h>
#include <GL/glew.h>

#include "../DrawValues.h"
#include "../../scene/structs/Scene.h"

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
    auto &cam = scene.cam;

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
      constexpr const ImVec2 windowSize = ImVec2(1000, 400);
      constexpr const ImVec2 windwoPos = ImVec2(WIDTH - windowSize.x - 10, 10);

      ImGui::SetNextWindowPos(windwoPos, ImGuiCond_Once);
      ImGui::SetNextWindowSize(windowSize, ImGuiCond_Once);

      ImGui::Begin("Controls");

      ImGui::SetNextItemOpen(true, ImGuiCond_Once);
      if (ImGui::CollapsingHeader("Zoom"))
      {
        float content_width = ImGui::GetContentRegionAvail().x;

        float slider_width = content_width * 0.6f;
        float input_width = content_width * 0.2f;

        {
          ImGui::Text("Cam FOV:");
          ImGui::SameLine();

          ImGui::PushItemWidth(slider_width);
          ImGui::SliderFloat("##fovSlider", &cam.camFOVdeg, 20.0f, 120.0f, "%0.1f");
          ImGui::PopItemWidth();

          ImGui::SameLine();

          ImGui::PushItemWidth(input_width);
          ImGui::InputFloat("##fovInput", &cam.camFOVdeg);
          ImGui::PopItemWidth();
        }
        {
          ImGui::Text("Cam Zoom:");
          ImGui::SameLine();

          ImGui::PushItemWidth(slider_width);
          ImGui::SliderFloat("##zoomSlider", &cam.camZoom, 0.1f, 4.0f, "%0.1f");
          ImGui::PopItemWidth();

          ImGui::SameLine();

          ImGui::PushItemWidth(input_width);
          ImGui::InputFloat("##zoomInput", &cam.camZoom);
          ImGui::PopItemWidth();
        }
      }

      ImGui::End();
    }
  }

  GuiWindow() = default;
};

extern GuiWindow guiWindow;