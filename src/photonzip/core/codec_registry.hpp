#pragma once

#include <string>
#include <vector>

#include "photonzip/core/codec_types.hpp"

namespace photonzip {

void register_codec(CodecVTable codec);
const CodecVTable& get_codec(const std::string& codec_name);
std::vector<std::string> list_codecs();
void register_builtin_codecs();

}  // namespace photonzip
