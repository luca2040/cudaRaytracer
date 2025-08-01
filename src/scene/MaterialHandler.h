#pragma once

#include <vector>
#include <algorithm>

#include "../math/Definitions.h"

class MaterialHandler
{
private:
public:
  std::vector<Material> materials;

  size_t mat(const Material &newMaterial)
  {
    auto i = std::find(materials.begin(), materials.end(), newMaterial);

    // Check if this material already exists
    if (i != materials.end())
    {
      return std::distance(materials.begin(), i);
    }

    // Add it to the list
    materials.push_back(newMaterial);
    return materials.size() - 1;
  }
};