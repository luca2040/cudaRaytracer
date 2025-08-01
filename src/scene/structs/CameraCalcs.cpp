#include "CameraCalcs.h"
#include "../../utils/gui/GuiWindow.h"
#include "../../math/Operations.h"
#include "../../utils/DrawValues.h"

void updateCameraRaygenData(Camera &cam)
{
  float cx = cos(cam.camXrot), sx = sin(cam.camXrot);
  float cy = cos(cam.camYrot), sy = sin(cam.camYrot);

  float3_L cameraDirVec = {0, sy, cy};

  mat3x3 yRotMat = {
      float3_L(cx, 0.0f, sx),
      float3_L(0.0f, 1.0f, 0.0f),
      float3_L(-sx, 0.0f, cx)};

  cam.camForward = yRotMat * cameraDirVec;
  cam.camRight = normalize(cross3(cam.camForward, {0, -1, 0})); // camForward, worldUp
  cam.camUp = cross3(cam.camRight, cam.camForward);

  // Camera settings declaration

  float camFOV = cam.camFOVdeg * M_PI / 180.0f;
  float imagePlaneHeight = 2.0f * tan(camFOV / 2.0f);
  float imagePlaneWidth = imagePlaneHeight * guiWindow.winDims.aspect;

  // Camera placement

  float3_L imageCenter = cam.camPos + cam.camForward;
  cam.imageX = cam.camRight * imagePlaneWidth * cam.camZoom;
  cam.imageY = cam.camUp * imagePlaneHeight * cam.camZoom;
  cam.camViewOrigin = imageCenter - cam.imageX * 0.5f - cam.imageY * 0.5f;

  cam.inverseWidthMinus = 1.0f / static_cast<float>(guiWindow.winDims.renderingWidth - 1);
  cam.inverseHeightMinus = 1.0f / static_cast<float>(guiWindow.winDims.renderingHeight - 1);
}