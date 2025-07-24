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
  template <typename T>
  inline void MakeEditableSlider(const char *title, const char *sliderName, const char *valueName,
                                 T &value, T sliderMin, T sliderMax,
                                 float sliderWidth, float valueWidth)
  {
    ImGui::TextUnformatted(title);
    ImGui::SameLine();

    ImGui::PushItemWidth(sliderWidth);

    if constexpr (std::is_same_v<T, float>)
      ImGui::SliderFloat(sliderName, &value, sliderMin, sliderMax, "%0.1f");
    else
      ImGui::SliderInt(sliderName, &value, sliderMin, sliderMax);

    ImGui::PopItemWidth();

    ImGui::SameLine();

    ImGui::PushItemWidth(valueWidth);

    if constexpr (std::is_same_v<T, float>)
      ImGui::InputFloat(valueName, &value);
    else
      ImGui::InputInt(valueName, &value);

    ImGui::PopItemWidth();
  }

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
      ImGui::Text("Rendering resolution: %i x %i", WIDTH, HEIGHT);

      ImGui::Separator();

      ImGui::Text("FPS: %.1f", io->Framerate);
      ImGui::Text("DeltaTime: %.4f s", io->DeltaTime);

      ImGui::Separator();

      ImGui::SetNextItemOpen(false, ImGuiCond_Once);
      if (ImGui::CollapsingHeader("Camera"))
      {
        ImGui::Text("Position: %0.1f, %0.1f, %0.1f", cam.camPos.x, cam.camPos.y, cam.camPos.z);
        ImGui::Text("Angle: %0.2f, %0.2f", cam.camXrot, cam.camYrot);
      }

      ImGui::SetNextItemOpen(false, ImGuiCond_Once);
      if (ImGui::CollapsingHeader("Scene"))
      {
        ImGui::Text("Vertices: %zu", scene.pointsCount);
        ImGui::Text("Triangles: %zu", scene.triangleNum);
        ImGui::Text("Dyn triangles: %zu", scene.dyntrianglesNum);
      }

      ImGui::End();
    }

    // Controls window
    {
      constexpr const ImVec2 windowSize = ImVec2(1000, 300);
      constexpr const ImVec2 windwoPos = ImVec2(WINDOW_WIDTH - windowSize.x - 10, 10);

      float contentWidth = ImGui::GetContentRegionAvail().x;
      float sliderWidth = contentWidth * 0.6f;
      float inputWidth = contentWidth * 0.2f;

      ImGui::SetNextWindowPos(windwoPos, ImGuiCond_Once);
      ImGui::SetNextWindowSize(windowSize, ImGuiCond_Once);

      ImGui::Begin("Controls");

      ImGui::SetNextItemOpen(false, ImGuiCond_Once);
      if (ImGui::CollapsingHeader("Debug"))
      {
        ImGui::Checkbox("Bounding box view mode", &scene.boundingBoxDebugView);
      }

      ImGui::SetNextItemOpen(false, ImGuiCond_Once);
      if (ImGui::CollapsingHeader("Camera"))
      {
        MakeEditableSlider<float>("Cam FOV:", "##fovSlider", "##fovInput",
                                  cam.camFOVdeg, 20.0f, 120.0f,
                                  sliderWidth, inputWidth);

        MakeEditableSlider<float>("Cam Zoom:", "##zoomSlider", "##zoomInput",
                                  cam.camZoom, 0.1f, 4.0f,
                                  sliderWidth, inputWidth);
      }

      ImGui::SetNextItemOpen(false, ImGuiCond_Once);
      if (ImGui::CollapsingHeader("Rendering"))
      {
        ImGui::Checkbox("Bounding box view mode", &scene.boundingBoxDebugView);

        MakeEditableSlider<int>("Max ray reflections:", "##rayReflectionsSlider", "##rayReflectionsInput",
                                scene.maxRayReflections, 0, 25,
                                sliderWidth, inputWidth);
      }

      ImGui::End();
    }
  }

  GuiWindow() = default;
};

extern GuiWindow guiWindow;