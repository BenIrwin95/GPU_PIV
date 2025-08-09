#include "standardHeader.hpp"

const std::string kernelSource_complexMaths = R"(
  float2 multiply_complex(float2 a, float2 b){
    float2 out;
    out.x = a.x*b.x - a.y*b.y;
    out.y = a.x*b.y + a.y*b.x;
    return out;
  }

  float2 add_complex(float2 a, float2 b){
    float2 out;
    out.x = a.x+b.x;
    out.y = a.y+b.y;
    return out;
  }

  float2 subtract_complex(float2 a, float2 b){
    float2 out;
    out.x = a.x-b.x;
    out.y = a.y-b.y;
    return out;
  }

  float2 scale_complex(float2 a, float scalar){
    float2 out;
    out.x = a.x*scalar;
    out.y = a.y*scalar;
    return out;
  }
)";
