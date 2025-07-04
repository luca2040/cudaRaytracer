#include "Shader.h"

GLuint compileShader(GLenum type, const char *src)
{
  GLuint shader = glCreateShader(type);
  glShaderSource(shader, 1, &src, nullptr);
  glCompileShader(shader);

  return shader;
}

GLuint createShaderProgram()
{
  GLuint vert = compileShader(GL_VERTEX_SHADER, vertexShaderSrc);
  GLuint frag = compileShader(GL_FRAGMENT_SHADER, fragmentShaderSrc);

  GLuint program = glCreateProgram();
  glAttachShader(program, vert);
  glAttachShader(program, frag);
  glLinkProgram(program);

  glDeleteShader(vert);
  glDeleteShader(frag);

  return program;
}

void setupQuad(GLuint &quadVAO, GLuint &quadVBO)
{
  float quadVertices[] = {
      // positions   // tex coords
      -1.0f, -1.0f, 0.0f, 0.0f, // bottom-left
      1.0f, -1.0f, 1.0f, 0.0f,  // bottom-right
      -1.0f, 1.0f, 0.0f, 1.0f,  // top-left
      1.0f, 1.0f, 1.0f, 1.0f    // top-right
  };

  glGenVertexArrays(1, &quadVAO);
  glGenBuffers(1, &quadVBO);

  glBindVertexArray(quadVAO);
  glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
  glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), quadVertices, GL_STATIC_DRAW);

  glEnableVertexAttribArray(0); // aPos
  glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void *)0);

  glEnableVertexAttribArray(1); // aTexCoord
  glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void *)(2 * sizeof(float)));

  glBindVertexArray(0);
}