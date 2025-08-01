#include "../../utils/DrawValues.h"
#include "../SceneBuilder.h"

#include "../builders/CubeBuilder.h"
#include "../builders/FlatSquareBuilder.h"

// ################ Transform functions ################

inline void normalRotation(u_int32_t time, float3_L &rotationAngles, float3_L &relativePos)
{
  float xrot = fmod((static_cast<float>(time) * 0.0005f), TWO_PI);
  float yrot = fmod((static_cast<float>(time) * 0.001f), TWO_PI);

  rotationAngles = {xrot, yrot, 0.0f};
}

inline void verticalShift(u_int32_t time, float3_L &rotationAngles, float3_L &relativePos)
{
  float timeValue = fmod((static_cast<float>(time) * 0.001f), TWO_PI);
  float ymov = std::sin(timeValue) * 0.8f;
  float xrot = std::cos(timeValue) * 0.2f;

  relativePos = {0.0f, ymov, 0.0f};
  rotationAngles = {0, xrot, 0};
}

// #####################################################

#define M(color, reflectiveness) materials.mat(Material(color, reflectiveness))

void composeScene()
{
  SceneBuilder builder;
  MaterialHandler &materials = builder.materials;

  builder.addObjectToScene(generateCube({1.0f, 0.0f, 2.0f}, 1.0f,
                                        normalRotation,
                                        M(0xFF0000, 0.8f), M(0x00FF00, 0.8f), M(0x0000FF, 0.8f), M(0xFFFF00, 0.8f), M(0xFF00FF, 0.8f), M(0x00FFFF, 0.8f)));

  builder.addObjectToScene(generateCube({-1.0f, 0.0f, 2.0f}, 1.0f,
                                        verticalShift,
                                        M(0xFF0000, 0.0f), M(0x00FF00, 0.0f), M(0x0000FF, 0.0f), M(0xFFFF00, 0.0f), M(0xFF00FF, 0.0f), M(0x00FFFF, 0.0f)));

  builder.addObjectToScene(generateFlatSquare({-2.5f, -2.5f, 3.5f}, // rear panel
                                              {2.5f, 2.5f, 3.5f},
                                              M(0xFFFFFF, 0.0f)));

  builder.addObjectToScene(generateFlatSquare({-2.5f, -2.5f, 0.0f}, // left panel
                                              {-2.5f, 2.5f, 3.5f},
                                              M(0xFFFFFF, 0.95f)));

  builder.addObjectToScene(generateFlatSquare({2.5f, -2.5f, 0.0f}, // right panel
                                              {2.5f, 2.5f, 3.5f},
                                              M(0xFFFFFF, 0.95f)));

  builder.addObjectToScene(generateFlatSquare({-2.5f, 2.5f, 0.0f}, // floor
                                              {2.5f, 2.5f, 3.5f},
                                              5,
                                              M(0xFFFFFF, 0.25f), M(0x000000, 0.2f)));

  builder.addObjectToScene(generateFlatSquare({-2.5f, -2.5f, 0.0f}, // top
                                              {2.5f, -2.5f, 3.5f},
                                              M(0x555555, 0.0f)));

  builder.compile();
}