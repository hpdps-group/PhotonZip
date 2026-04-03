#pragma once

#include <stdexcept>
#include <string>

namespace photonzip {

class Error : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};

inline Error make_unknown_codec_error(const std::string& codec_name) {
  return Error("Unknown codec: " + codec_name);
}

}  // namespace photonzip
