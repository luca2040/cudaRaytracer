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

// Yes this is bad for readability but whatever
#define M(color, diffuse) materials.mat(Material(color, diffuse, 0, 0.0f, 0.0f))
#define MR(color, diffuse, reflectiveness) materials.mat(Material(color, diffuse, 0, 0.0f, reflectiveness))
#define M2(color, diffuse, emCol, emStren) materials.mat(Material(color, diffuse, emCol, emStren, 0.0f))

void composeScene()
{
  SceneBuilder builder;
  MaterialHandler &materials = builder.materials;

  builder.addObjectToScene(generateCube({1.0f, 2.0f, 2.0f}, 1.0f,
                                        {0, M_PI_4, 0},
                                        MR(0xFF0000, 0.5f, 0.8f))); // Red cube (reflective)

  builder.addObjectToScene(generateCube({-1.0f, 1.63f, 2.0f}, 1.0f,
                                        {M_PI_4, M_PI_4, 0},
                                        M(0x00FF00, 0.5f))); // Green cube

  builder.addObjectToScene(generateFlatSquare({-2.5f, -2.5f, 3.5f}, // rear panel
                                              {2.5f, 2.5f, 3.5f},
                                              M(0xFFFFFF, 0.2f)));

  builder.addObjectToScene(generateFlatSquare({-2.5f, -2.5f, 0.0f}, // left panel
                                              {-2.5f, 2.5f, 3.5f},
                                              MR(0xFFFFFF, 1.0f, 0.8f)));

  builder.addObjectToScene(generateFlatSquare({2.5f, -2.5f, 0.0f}, // right panel
                                              {2.5f, 2.5f, 3.5f},
                                              MR(0xFFFFFF, 1.0f, 0.8f)));

  builder.addObjectToScene(generateFlatSquare({-2.5f, 2.5f, 0.0f}, // floor
                                              {2.5f, 2.5f, 3.5f},
                                              5,
                                              M(0xFFFFFF, 0.5f), M(0x000000, 0.5f)));

  builder.addObjectToScene(generateFlatSquare({-2.5f, -2.5f, 0.0f}, // top
                                              {2.5f, -2.5f, 3.5f},
                                              M(0xAAAAAA, 0.5f)));

  builder.addObjectToScene(generateFlatSquare({-1.0f, -2.25f, 1.5f}, // top ligth
                                              {1.0f, -2.25f, 2.0f},
                                              M2(0xFFFFFF, 0.5f, 0xFFFFFF, 10.0f)));

  builder.compile();
}