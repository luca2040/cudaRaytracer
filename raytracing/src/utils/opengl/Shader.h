#pragma once

#include <GL/glew.h>

inline const char *vertexShaderSrc = R"(
#version 330 core
layout(location = 0) in vec2 aPos;       // vertex position
layout(location = 1) in vec2 aTexCoord;  // texture coordinate

out vec2 TexCoord;

void main()
{
    gl_Position = vec4(aPos.xy, 0.0, 1.0);
    TexCoord = aTexCoord;
}
)";

inline const char *fragmentShaderSrc = R"(
#version 330 core
in vec2 TexCoord;
out vec4 FragColor;

uniform sampler2D tex;

void main()
{
    FragColor = texture(tex, TexCoord);
}
)";

GLuint createShaderProgram();
void setupQuad(GLuint &quadVAO, GLuint &quadVBO);