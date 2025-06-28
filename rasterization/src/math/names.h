#ifndef NAMES_STRUCTS
#define NAMES_STRUCTS

struct int2_L
{
  int x;
  int y;

  int2_L() = default;
  int2_L(int x, int y) : x(x), y(y) {}
};

struct triangleidx
{
  unsigned long v1;
  unsigned long v2;
  unsigned long v3;

  int col;

  triangleidx() = default;
  triangleidx(unsigned long v1, unsigned long v2, unsigned long v3, int col)
      : v1(v1), v2(v2), v3(v3), col(col) {}
};

struct float2_L
{
  float x;
  float y;

  float2_L() = default;
  float2_L(float x, float y) : x(x), y(y) {}

  inline float2_L &operator+=(const float2_L &rhs) noexcept
  {
    x += rhs.x;
    y += rhs.y;
    return *this;
  }

  inline float2_L &operator-=(const float2_L &rhs) noexcept
  {
    x -= rhs.x;
    y -= rhs.y;
    return *this;
  }
};

inline float2_L operator+(const float2_L &a, const float2_L &b) noexcept
{
  return {a.x + b.x, a.y + b.y};
}

inline float2_L operator-(const float2_L &a, const float2_L &b) noexcept
{
  return {a.x - b.x, a.y - b.y};
}

inline float dot(const float2_L &a, const float2_L &b) noexcept
{
  return a.x * b.x + a.y * b.y;
}

struct float3_L
{
  float x;
  float y;
  float z;

  float3_L() = default;
  float3_L(float x, float y, float z) : x(x), y(y), z(z) {}

  inline float3_L &operator+=(const float3_L &rhs) noexcept
  {
    x += rhs.x;
    y += rhs.y;
    z += rhs.z;
    return *this;
  }

  inline float3_L &operator-=(const float3_L &rhs) noexcept
  {
    x -= rhs.x;
    y -= rhs.y;
    z -= rhs.z;
    return *this;
  }

  inline operator float2_L() const
  {
    return float2_L{x, y};
  }
};

inline float3_L operator+(const float3_L &a, const float3_L &b) noexcept
{
  return {a.x + b.x, a.y + b.y, a.z + b.z};
}

inline float3_L operator-(const float3_L &a, const float3_L &b) noexcept
{
  return {a.x - b.x, a.y - b.y, a.z - b.z};
}

struct mat3x3
{
  float3_L rows[3];

  inline const float3_L &operator[](int i) const { return rows[i]; }
  inline float3_L &operator[](int i) { return rows[i]; }

  inline float3_L operator*(const float3_L &v) const
  {
    return {
        rows[0].x * v.x + rows[0].y * v.y + rows[0].z * v.z,
        rows[1].x * v.x + rows[1].y * v.y + rows[1].z * v.z,
        rows[2].x * v.x + rows[2].y * v.y + rows[2].z * v.z};
  }

  inline mat3x3 operator*(const mat3x3 &b) const
  {
    mat3x3 r;
    for (int i = 0; i < 3; ++i)
    {
      r.rows[i] = {
          rows[i].x * b[0].x + rows[i].y * b[1].x + rows[i].z * b[2].x,
          rows[i].x * b[0].y + rows[i].y * b[1].y + rows[i].z * b[2].y,
          rows[i].x * b[0].z + rows[i].y * b[1].z + rows[i].z * b[2].z};
    }
    return r;
  }
};

#endif