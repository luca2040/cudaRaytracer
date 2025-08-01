#pragma once

#include <iostream>
#include <SDL2/SDL.h>
#include <GL/glew.h>

#include "../DrawValues.h"
#include "../../scene/structs/Scene.h"

#include "../../third_party/imgui/imgui.h"
// #include "../../third_party/imgui/imgui_impl_sdl2.h"
// #include "../../third_party/imgui/imgui_impl_opengl3.h"

struct windowDimensions
{
  int windowWidth;
  int windowHeight;

  int renderingWidth;
  int renderingHeight;

  float renderingScale;

  // Referred to rendering resolution
  float halfWidth;
  float halfHeight;

  float aspect;

  inline void updateValues()
  {
    renderingWidth = windowWidth * renderingScale;
    renderingHeight = windowHeight * renderingScale;

    halfWidth = static_cast<float>(renderingWidth) * 0.5f;
    halfHeight = static_cast<float>(renderingHeight) * 0.5f;

    aspect = static_cast<float>(renderingWidth) / static_cast<float>(renderingHeight);
  }

  windowDimensions() = default;
  windowDimensions(int winWidth, int winHeight, float scale) : windowWidth(winWidth), windowHeight(winHeight), renderingScale(scale)
  {
    updateValues();
  }
};

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
  bool resChanged = false; // Variable used to indicate when a change to the resolution has been made

  ImGuiIO *io = nullptr;

  const GLubyte *openGLversion;
  const GLubyte *openGLrenderer;

  windowDimensions winDims = {3840, 2160, 0.5f};

  inline void RenderGui()
  {
    auto &cam = scene->cam;

    // Status window
    {
      ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_Once);
      ImGui::SetNextWindowSize(ImVec2(800, 400), ImGuiCond_Once);

      ImGui::Begin("Status");

      ImGui::Text("OpenGL version: %s", openGLversion);
      ImGui::Text("%s", openGLrenderer);
      ImGui::Text("Rendering resolution: %i x %i", winDims.renderingWidth, winDims.renderingHeight);
      if (ImGui::SliderFloat("##ResolutionMultiplier", &winDims.renderingScale, 0.1f, 1.0f, "%.01f"))
      {
        resChanged = true;
      }

      ImGui::Separator();

      ImGui::Text("FPS: %.1f", io->Framerate);
      ImGui::Text("Frame time: %.4f ms", io->DeltaTime * 1000.0f);

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
      const ImVec2 windowSize = ImVec2(1000, 300);
      const ImVec2 windowPos = ImVec2(winDims.windowWidth - windowSize.x - 10, 10);

      ImGui::SetNextWindowPos(windowPos, ImGuiCond_Once);
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