void sumArrays(float in[][3], int rows, float *val, float mult)
{
  for (int i = 0; i < rows; i++)
  {
    in[i][0] += (val[0] * mult);
    in[i][1] += (val[1] * mult);
    in[i][2] += (val[2] * mult);
  }
}

void matMult(float *in, float mat[3][3])
{
  float result[3];

  result[0] = in[0] * mat[0][0] + in[1] * mat[1][0] + in[2] * mat[2][0];
  result[1] = in[0] * mat[0][1] + in[1] * mat[1][1] + in[2] * mat[2][1];
  result[2] = in[0] * mat[0][2] + in[1] * mat[1][2] + in[2] * mat[2][2];

  in[0] = result[0];
  in[1] = result[1];
  in[2] = result[2];
}
