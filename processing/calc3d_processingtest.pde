// Camera settings
float camZ = 600;

// Rotations
float[] rotcenter = {0, 0, 3000};
float yrot = 0;
float xrot = 0;

boolean autorotating = true;
boolean iscube = true;

// Points

// Cube
float[][] pointscube = {
  // Front face
  {-1000, -1000, 2000},
  {1000, -1000, 2000},
  {1000, 1000, 2000},
  {-1000, 1000, 2000},
  // Back face
  {-1000, -1000, 4000},
  {1000, -1000, 4000},
  {1000, 1000, 4000},
  {-1000, 1000, 4000}
};

// Pyramid
float[][] pointspyr = {
  // Front face
  {-1000, -1000, 2000},
  {1000, -1000, 2000},
  {1000, 1000, 2000},
  {-1000, 1000, 2000},
  // Back vertex
  {0, 0, 4000}
};

// Borders

// Cube
int[][] connectionscube = {
  {0, 1},
  {1, 2},
  {2, 3},
  {3, 0},

  {4, 5},
  {5, 6},
  {6, 7},
  {7, 4},

  {0, 4},
  {1, 5},
  {2, 6},
  {3, 7}
};

// Pyramid
int[][] connectionspyr = {
  {0, 1},
  {1, 2},
  {2, 3},
  {3, 0},

  {0, 4},
  {1, 4},
  {2, 4},
  {3, 4},
};

void setup() {
  size(1000, 1000);
  frameRate(144);

  surface.setTitle("Test - 3D");
}

void draw() {
  // Processing's needed things

  background(0);
  stroke(255);

  if (autorotating) {
    xrot = (millis() / 2000f) % TWO_PI;
    yrot = (millis() / 1000f) % TWO_PI;
  }

  // Copy array

  float[][] actualpoints = (iscube ? pointscube : pointspyr);

  float[][] moddedpoints = new float[actualpoints.length][];
  for (int i = 0; i < actualpoints.length; i++) {
    moddedpoints[i] = actualpoints[i].clone();
  }

  // Subtract center from every point to set the rotation center

  sumArrays(moddedpoints, rotcenter, -1);

  // Apply y and x rotation

  float[][] yrotmat = {
    {cos(yrot), 0, sin(yrot)},
    {0, 1, 0},
    {-sin(yrot), 0, cos(yrot)}
  };

  float[][] xrotmat = {
    {1, 0, 0},
    {0, cos(xrot), -sin(xrot)},
    {0, sin(xrot), cos(xrot)}
  };

  for (int i = 0; i < moddedpoints.length; i++) {
    matMult(moddedpoints[i], yrotmat);
    matMult(moddedpoints[i], xrotmat);
  }

  // Re-add center from every point to reposition the points but rotated

  sumArrays(moddedpoints, rotcenter, 1);

  // Calc and draw points

  float[][] points2d = new float[moddedpoints.length][2];

  for (int i = 0; i < moddedpoints.length; i++) {
    float xr = (camZ * moddedpoints[i][0]) / (camZ + moddedpoints[i][2]);
    float yr = (camZ * moddedpoints[i][1]) / (camZ + moddedpoints[i][2]);

    // Shift for the Processing window
    // Not important for the calculations
    xr += width / 2;
    yr += height / 2;

    points2d[i][0] = xr;
    points2d[i][1] = yr;
  }

  // Draw borders

  int[][] actualconnections = (iscube ? connectionscube : connectionspyr);

  for (int i = 0; i < actualconnections.length; i++) {
    float x1 = points2d[actualconnections[i][0]][0];
    float y1 = points2d[actualconnections[i][0]][1];

    float x2 = points2d[actualconnections[i][1]][0];
    float y2 = points2d[actualconnections[i][1]][1];

    line(x1, y1, x2, y2);
  }
}

// Input handling

void mouseDragged() {
  int deltax = mouseX - pmouseX;
  int deltay = mouseY - pmouseY;

  yrot += float(deltax) / 200;
  xrot -= float(deltay) / 200;

  yrot %= TWO_PI;
  xrot %= TWO_PI;
}

void keyReleased() {
  if (key == 'a') {
    autorotating = !autorotating;
  } else if (key == 'c') {
    iscube = !iscube;
  }
}

// General math functions

void sumArrays(float[][] in, float[] val, float mult) {
  for (int i = 0; i < in.length; i++) {
    in[i][0] += (val[0] * mult);
    in[i][1] += (val[1] * mult);
    in[i][2] += (val[2] * mult);
  }
}

void matMult(float[] in, float[][] mat) {
  float[] result = in.clone();

  result[0] = in[0] * mat[0][0] + in[1] * mat[1][0] + in[2] * mat[2][0];
  result[1] = in[0] * mat[0][1] + in[1] * mat[1][1] + in[2] * mat[2][1];
  result[2] = in[0] * mat[0][2] + in[1] * mat[1][2] + in[2] * mat[2][2];

  in[0] = result[0];
  in[1] = result[1];
  in[2] = result[2];
}
