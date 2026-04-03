#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "photonzip/core/codec_types.hpp"

namespace photonzip::codec_value {

inline const CodecDict& expect_dict(const CodecValue& value, const char* what) {
  const auto* dict = std::get_if<std::shared_ptr<CodecDict>>(&value.value);
  if (!dict || !(*dict)) {
    throw std::runtime_error(std::string("Expected a dict for ") + what + ".");
  }
  return *(*dict);
}

inline const CodecList& expect_list(const CodecValue& value, const char* what) {
  const auto* list = std::get_if<std::shared_ptr<CodecList>>(&value.value);
  if (!list || !(*list)) {
    throw std::runtime_error(std::string("Expected a list for ") + what + ".");
  }
  return *(*list);
}

inline std::int64_t expect_int(const CodecValue& value, const char* what) {
  if (const auto* int_value = std::get_if<std::int64_t>(&value.value)) {
    return *int_value;
  }
  throw std::runtime_error(std::string("Expected an integer for ") + what + ".");
}

inline double expect_double(const CodecValue& value, const char* what) {
  if (const auto* double_value = std::get_if<double>(&value.value)) {
    return *double_value;
  }
  if (const auto* int_value = std::get_if<std::int64_t>(&value.value)) {
    return static_cast<double>(*int_value);
  }
  throw std::runtime_error(std::string("Expected a number for ") + what + ".");
}

inline bool expect_bool(const CodecValue& value, const char* what) {
  if (const auto* bool_value = std::get_if<bool>(&value.value)) {
    return *bool_value;
  }
  throw std::runtime_error(std::string("Expected a bool for ") + what + ".");
}

inline const CodecValue* find_value(const CodecDict& dict, const char* key) {
  const auto it = dict.find(key);
  return it == dict.end() ? nullptr : &it->second;
}

template <typename T, typename Parser>
inline T parse_scalar(const CodecDict& dict, const char* key, T fallback, Parser parser) {
  const CodecValue* value = find_value(dict, key);
  if (!value) {
    return fallback;
  }
  return parser(*value, key);
}

inline std::vector<double> parse_double_list(const CodecDict& dict,
                                             const char* key,
                                             std::vector<double> fallback) {
  const CodecValue* value = find_value(dict, key);
  if (!value) {
    return fallback;
  }

  std::vector<double> out;
  for (const auto& item : expect_list(*value, key)) {
    out.push_back(expect_double(item, key));
  }
  return out;
}

inline std::vector<std::uint32_t> parse_u32_list(const CodecDict& dict,
                                                 const char* key,
                                                 std::vector<std::uint32_t> fallback) {
  const CodecValue* value = find_value(dict, key);
  if (!value) {
    return fallback;
  }

  std::vector<std::uint32_t> out;
  for (const auto& item : expect_list(*value, key)) {
    const auto parsed = expect_int(item, key);
    if (parsed < 0) {
      throw std::runtime_error(std::string("Expected non-negative values for ") + key + ".");
    }
    out.push_back(static_cast<std::uint32_t>(parsed));
  }
  return out;
}

}  // namespace photonzip::codec_value
