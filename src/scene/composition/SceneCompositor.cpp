#include "../../utils/DrawValues.h"
#include "../SceneBuilder.h"

#include "../builders/CubeBuilder.h"
#include "../builders/SphereBuilder.h"
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
#define M(color, diffuse) materials.mat(Material(color, diffuse, 0, 0.0f, 0.0f, false))
#define MR(color, reflectiveness, isMetal) materials.mat(Material(color, 1.0f, 0, 0.0f, reflectiveness, isMetal))
#define ME(color, diffuse, emCol, emStren) materials.mat(Material(color, diffuse, emCol, emStren, 0.0f, false))

void composeScene()
{
  SceneBuilder builder;
  MaterialHandler &materials = builder.materials;

  // [TODO] Gotta clean up this mess with an integrated editor

  // builder.addObjectToScene(generateCube({1.0f, 2.0f, 2.0f}, 1.0f,
  //                                       {0, M_PI_4, 0},
  //                                       MR(0xFF0000, 0.5f, 0.8f))); // Red cube (reflective)

  // builder.addObjectToScene(generateCube({-1.0f, 1.63f, 2.0f}, 1.0f,
  //                                       {M_PI_4, M_PI_4, 0},
  //                                       M(0x00FF00, 0.5f))); // Green cube

  builder.addObjectToScene(generateCube({-1.0f, 2.0f, 2.0f}, 1.0f,
                                        {0, 0.38f, 0},
                                        M(0xFFFF00, 0.75f))); // Green cube V2

  // builder.addObjectToScene(generateSphere({-0.95f, 1.7f, 2.0f}, 0.8f,
  //                                         {0, 0, 0},
  //                                         MR(0xFFFFFF, 1.0f))); // Mirror sphere 1

  // builder.addObjectToScene(generateSphere({0.95f, 1.7f, 2.0f}, 0.8f,
  //                                         {0, 0, 0},
  //                                         MR(0xFFFFFF, 1.0f))); // Mirror sphere 2

  builder.addObjectToScene(generateSphere({0.95f, 1.35f, 2.0f}, 1.15f,
                                          {0, 0, 0},
                                          MR(0xFF0000, 0.75f, true))); // Big shpere 2

  builder.addObjectToScene(generateFlatSquare({-2.5f, -2.5f, 3.5f}, // rear panel
                                              {2.5f, 2.5f, 3.5f},
                                              M(0xFFFFFF, 0.5f)));

  builder.addObjectToScene(generateFlatSquare({-2.5f, -2.5f, 0.0f}, // left panel
                                              {-2.5f, 2.5f, 3.5f},
                                              M(0x1AC721, 0.5f)));

  builder.addObjectToScene(generateFlatSquare({2.5f, -2.5f, 0.0f}, // right panel
                                              {2.5f, 2.5f, 3.5f},
                                              M(0x0000DD, 0.5f)));

  // builder.addObjectToScene(generateFlatSquare({-2.5f, 2.5f, 0.0f}, // floor
  //                                             {2.5f, 2.5f, 3.5f},
  //                                             5,
  //                                             M(0x1AC721, 0.5f), M(0x18851B, 0.5f)));

  builder.addObjectToScene(generateFlatSquare({-2.5f, 2.5f, 0.0f}, // Black and white reflective floor
                                              {2.5f, 2.5f, 3.5f},
                                              5,
                                              MR(0x000000, 0.5f, false), MR(0xFFFFFF, 0.5f, false)));

  builder.addObjectToScene(generateFlatSquare({-2.5f, -2.5f, 0.0f}, // top
                                              {2.5f, -2.5f, 3.5f},
                                              M(0xAAAAAA, 0.5f)));

  builder.addObjectToScene(generateFlatSquare({-1.0f, -2.25f, 1.5f}, // top ligth
                                              {1.0f, -2.25f, 2.0f},
                                              ME(0xFFFFFF, 0.5f, 0xFFFFFF, 10.0f)));

  builder.compile();
}