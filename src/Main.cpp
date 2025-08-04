#include <iostream>
#include <SDL2/SDL.h>
#include <GL/glew.h>
#include <cuda_runtime.h>

#include "utils/DrawLoop.h"
#include "utils/gui/GuiWindow.h"
#include "math/Definitions.h"

#include "third_party/imgui/imgui.h"
#include "third_party/imgui/imgui_impl_sdl2.h"
#include "third_party/imgui/imgui_impl_opengl3.h"

#include "utils/Profiling.h"

int main()
{
  std::cout << "Starting the best 3D renderer ever - Now RTX!" << std::endl;

  if (SDL_Init(SDL_INIT_VIDEO) < 0)
  {
    std::cerr << "SDL could not initialize: " << SDL_GetError() << std::endl;
    return 1;
  }

  SDL_GL_SetAttribute(SDL_GL_CONTEXT_FLAGS, 0);
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 0);

  SDL_SetHint(SDL_HINT_IME_SHOW_UI, "1");

  SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
  SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);
  SDL_GL_SetAttribute(SDL_GL_STENCIL_SIZE, 8);

  // Allocate scene

  cudaHostAlloc(&scene, sizeof(Scene), cudaHostAllocDefault);
  Scene newScene;
  *scene = newScene;

  // Imgui settings

  float main_scale = ImGui_ImplSDL2_GetContentScaleForDisplay(0);

  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  guiWindow.io = &ImGui::GetIO();
  (void)guiWindow.io;
  guiWindow.io->ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

  ImGui::StyleColorsLight();

  ImGuiStyle &style = ImGui::GetStyle();
  style.ScaleAllSizes(main_scale);
  style.FontScaleDpi = main_scale;
  style.TabRounding = 8.0f;
  style.FrameRounding = 8.0f;
  style.GrabRounding = 8.0f;
  style.WindowRounding = 8.0f;
  style.PopupRounding = 8.0f;
  style.FontSizeBase = 15.0f;

  SDL_Window *window = SDL_CreateWindow("The Best 3D Renderer Ever - RTX edition",
                                        SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
                                        guiWindow.winDims.windowWidth, guiWindow.winDims.windowHeight, SDL_WINDOW_OPENGL | SDL_WINDOW_SHOWN | SDL_WINDOW_RESIZABLE);

  SDL_GLContext glContext = SDL_GL_CreateContext(window);

  if (!glContext)
  {
    std::cerr << "OpenGL context creation failed: " << SDL_GetError() << std::endl;
    return 1;
  }

  SDL_GL_MakeCurrent(window, glContext);

  glViewport(0, 0, guiWindow.winDims.windowWidth, guiWindow.winDims.windowHeight);

  if (glewInit() != GLEW_OK)
  {
    std::cerr << "GLEW failed to initialize" << std::endl;
    return -1;
  }

  // Check opengl version
  guiWindow.openGLversion = glGetString(GL_VERSION);
  guiWindow.openGLrenderer = glGetString(GL_RENDERER);

  GLint defaultPbo;
  GLuint renderingPbo, tex;

  glGetIntegerv(GL_PIXEL_UNPACK_BUFFER_BINDING, &defaultPbo);

  glGenBuffers(1, &renderingPbo);
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, renderingPbo);
  glBufferData(GL_PIXEL_UNPACK_BUFFER, guiWindow.winDims.renderingWidth * guiWindow.winDims.renderingHeight * 4, nullptr, GL_STREAM_DRAW);
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, defaultPbo);

  glGenTextures(1, &tex);
  glBindTexture(GL_TEXTURE_2D, tex);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, guiWindow.winDims.renderingWidth, guiWindow.winDims.renderingHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

  SDL_GL_SetSwapInterval(0); // Disable vsync

  ImGui_ImplSDL2_InitForOpenGL(window, glContext);
  ImGui_ImplOpenGL3_Init();

  SDL_Event event;
  bool running = true;

  int2_L mouse(0, 0);
  int2_L pMouse(0, 0);

  onSceneComposition();
  onSetupFrame(renderingPbo);

  guiWindow.resChanged = true;

  while (running)
  {
    ZONESCOPEDNC("Main SDL while cycle", PROFILER_ORANGE);

    while (SDL_PollEvent(&event))
    {
      ImGui_ImplSDL2_ProcessEvent(&event);
      switch (event.type)
      {
      case SDL_QUIT:
        running = false;
        break;

      case SDL_KEYDOWN:
        if (event.key.keysym.sym == SDLK_ESCAPE)
          running = false;

        keyPressed(event.key.keysym.sym);
        break;

      case SDL_MOUSEMOTION:
        pMouse = mouse;
        mouse = int2_L(event.motion.x, event.motion.y);

        mouseMoved(mouse, pMouse);
        break;

      case SDL_WINDOWEVENT:
        if (event.window.event == SDL_WINDOWEVENT_RESIZED)
        {
          guiWindow.winDims.windowWidth = event.window.data1;
          guiWindow.winDims.windowHeight = event.window.data2;

          guiWindow.resChanged = true;
        }
        break;
      }
    }

    if (guiWindow.resChanged)
    {
      guiWindow.resChanged = false;
      guiWindow.winDims.updateValues();

      glViewport(0, 0, guiWindow.winDims.windowWidth, guiWindow.winDims.windowHeight);

      glBindBuffer(GL_PIXEL_UNPACK_BUFFER, renderingPbo);
      glBufferData(GL_PIXEL_UNPACK_BUFFER, guiWindow.winDims.renderingWidth * guiWindow.winDims.renderingHeight * 4, nullptr, GL_STREAM_DRAW);

      glBindTexture(GL_TEXTURE_2D, tex);
      glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, guiWindow.winDims.renderingWidth, guiWindow.winDims.renderingHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);

      size_t accumulationSize = guiWindow.winDims.renderingWidth * guiWindow.winDims.renderingHeight * sizeof(float3);

      if (scene->d_accumulationBuffer)
        cudaFree(scene->d_accumulationBuffer);
      cudaMalloc(&scene->d_accumulationBuffer, accumulationSize);
      cudaMemset(scene->d_accumulationBuffer, 0, accumulationSize); // Zero it, the value is not important since it'll get overwritten if scene->accumulate == false
    }

    checkForKeys();

    // ###############################
    // Testing imgui

    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplSDL2_NewFrame();
    ImGui::NewFrame();

    guiWindow.RenderGui();

    // Bind rendering pbo, render scene, rebind default pbo
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, renderingPbo);
    drawFrame(tex, renderingPbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, defaultPbo);

    ImGui::Render();

    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    SDL_GL_SwapWindow(window);
  }

  onClose();

  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplSDL2_Shutdown();
  ImGui::DestroyContext();

  SDL_DestroyWindow(window);
  SDL_Quit();

  return 0;
}