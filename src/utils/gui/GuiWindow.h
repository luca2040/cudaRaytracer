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
  inline void MakeEditableSlider(const char *label, T &value, T sliderMin, T sliderMax)
  {
    ImGui::PushID(label); // Avoid id conflicts

    // Title
    ImGui::TextUnformatted(label);

    if (ImGui::BeginTable("slider_table", 2, ImGuiTableFlags_SizingStretchSame))
    {
      ImGui::TableSetupColumn("Slider", ImGuiTableColumnFlags_WidthStretch);
      ImGui::TableSetupColumn("Input", ImGuiTableColumnFlags_WidthStretch);

      ImGui::TableNextRow();

      // Slider
      ImGui::TableSetColumnIndex(0);
      ImGui::PushItemWidth(-FLT_MIN);
      if constexpr (std::is_same_v<T, float>)
        ImGui::SliderFloat("##slider", &value, sliderMin, sliderMax, "%.1f");
      else
        ImGui::SliderInt("##slider", &value, sliderMin, sliderMax);

      // Input
      ImGui::TableSetColumnIndex(1);
      ImGui::PushItemWidth(-FLT_MIN);
      if constexpr (std::is_same_v<T, float>)
        ImGui::InputFloat("##input", &value);
      else
        ImGui::InputInt("##input", &value);

      ImGui::EndTable();
    }

    ImGui::PopID();
  }

public:
  ImGuiIO *io = nullptr;
  const GLubyte *openGLversion;

  inline void RenderGui()
  {
    auto &cam = scene->cam;

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
        ImGui::Text("Vertices: %zu", scene->pointsCount);
        ImGui::Text("Triangles: %zu", scene->triangleNum);
        ImGui::Text("Models: %zu", scene->sceneobjectsNum);
      }

      ImGui::End();
    }

    // Controls window
    {
      constexpr const ImVec2 windowSize = ImVec2(1000, 300);
      constexpr const ImVec2 windwoPos = ImVec2(WINDOW_WIDTH - windowSize.x - 10, 10);

      ImGui::SetNextWindowPos(windwoPos, ImGuiCond_Once);
      ImGui::SetNextWindowSize(windowSize, ImGuiCond_Once);

      ImGui::Begin("Controls");

      ImGui::SetNextItemOpen(false, ImGuiCond_Once);
      if (ImGui::CollapsingHeader("Debug"))
      {
        ImGui::Checkbox("Cuda transform kernel sync", &scene->transformSync);
        ImGui::Checkbox("Cuda after trace sync", &scene->afterTraceSync);
        ImGui::Checkbox("Bounding box view mode", &scene->boundingBoxDebugView);
      }

      ImGui::SetNextItemOpen(false, ImGuiCond_Once);
      if (ImGui::CollapsingHeader("Camera"))
      {
        MakeEditableSlider("Cam FOV", cam.camFOVdeg, 20.0f, 120.0f);
        MakeEditableSlider("Cam Zoom", cam.camZoom, 0.1f, 4.0f);
      }

      ImGui::SetNextItemOpen(false, ImGuiCond_Once);
      if (ImGui::CollapsingHeader("Rendering"))
      {
        ImGui::ColorEdit3("Background color", &scene->backgroundColor.x);
        MakeEditableSlider("Max ray reflections", scene->maxRayReflections, 0, 100);
      }

      ImGui::End();
    }
  }

  GuiWindow() = default;
};

extern GuiWindow guiWindow;