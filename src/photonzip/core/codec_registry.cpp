#include "photonzip/core/codec_registry.hpp"

#include <mutex>
#include <unordered_map>
#include <utility>

#include "photonzip/codecs/mans/mans_codec.hpp"
#include "photonzip/core/errors.hpp"

namespace photonzip {
namespace {

using Registry = std::unordered_map<std::string, CodecVTable>;

Registry& codec_registry() {
  static Registry registry;
  return registry;
}

std::mutex& codec_registry_mutex() {
  static std::mutex mutex;
  return mutex;
}

std::once_flag& builtin_registration_flag() {
  static std::once_flag flag;
  return flag;
}

}  // namespace

void register_codec(CodecVTable codec) {
  std::lock_guard<std::mutex> lock(codec_registry_mutex());
  codec_registry()[codec.name] = std::move(codec);
}

const CodecVTable& get_codec(const std::string& codec_name) {
  register_builtin_codecs();

  std::lock_guard<std::mutex> lock(codec_registry_mutex());
  const auto it = codec_registry().find(codec_name);
  if (it == codec_registry().end()) {
    throw make_unknown_codec_error(codec_name);
  }
  return it->second;
}

std::vector<std::string> list_codecs() {
  register_builtin_codecs();

  std::lock_guard<std::mutex> lock(codec_registry_mutex());
  std::vector<std::string> names;
  names.reserve(codec_registry().size());
  for (const auto& item : codec_registry()) {
    names.push_back(item.first);
  }
  return names;
}

void register_builtin_codecs() {
  std::call_once(builtin_registration_flag(), []() {
    register_codec(make_mans_codec());
  });
}

}  // namespace photonzip
