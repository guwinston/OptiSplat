#include "load.h"
#include "render.h"
#include "utils.h"

#include <Eigen/Dense>

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <iomanip>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#if defined(_WIN32)
#  include <windows.h>   // LoadLibrary / GetProcAddress / FreeLibrary
#else
#  include <dlfcn.h>     // dlopen / dlsym / dlclose
#endif

#include <nlohmann/json.hpp>


namespace optisplat {

// ======================================================================
// Explicit template instantiations
// ======================================================================
template class PlyLoader<0>;
template class PlyLoader<1>;
template class PlyLoader<2>;
template class PlyLoader<3>;

template class SogLoader<0>;
template class SogLoader<1>;
template class SogLoader<2>;
template class SogLoader<3>;

template class LoaderFactory<0>;
template class LoaderFactory<1>;
template class LoaderFactory<2>;
template class LoaderFactory<3>;


// ======================================================================
// Internal helpers (anonymous namespace -- not exposed outside this TU)
// ======================================================================
namespace {

// ----------------------------------------------------------------------
// String / path utilities
// ----------------------------------------------------------------------
inline std::string toLowerCopy(std::string text) {
    std::transform(text.begin(), text.end(), text.begin(),
        [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return text;
}

inline bool hasExtension(const std::string& path, const char* ext) {
    std::filesystem::path fp(path);
    return toLowerCopy(fp.extension().string()) == ext;
}


// ----------------------------------------------------------------------
// CRC-32
// ----------------------------------------------------------------------
std::array<uint32_t, 256> buildCrc32Table() {
    std::array<uint32_t, 256> table{};
    for (uint32_t n = 0; n < 256; ++n) {
        uint32_t c = n;
        for (int k = 0; k < 8; ++k)
            c = (c & 1U) ? (0xEDB88320U ^ (c >> 1U)) : (c >> 1U);
        table[n] = c;
    }
    return table;
}

uint32_t crc32Compute(const uint8_t* data, size_t size) {
    static const std::array<uint32_t, 256> table = buildCrc32Table();
    uint32_t bits = 0xFFFFFFFFU;
    for (size_t i = 0; i < size; ++i)
        bits = (bits >> 8U) ^ table[(bits ^ data[i]) & 0xFFU];
    return bits ^ 0xFFFFFFFFU;
}


// ----------------------------------------------------------------------
// ZIP structures
// ----------------------------------------------------------------------
constexpr uint32_t kSogCodebookSize = 256;
constexpr uint8_t  kQuatTagBase     = 252;
constexpr float    kQuantEpsilon    = 1e-8f;

#pragma pack(push, 1)
struct ZipLocalHeader {
    uint32_t signature         = 0x04034b50;
    uint16_t versionNeeded     = 20;
    uint16_t flags             = 0x0808;
    uint16_t method            = 0;
    uint16_t modTime           = 0;
    uint16_t modDate           = 0;
    uint32_t crc32             = 0;
    uint32_t compressedSize    = 0;
    uint32_t uncompressedSize  = 0;
    uint16_t nameLen           = 0;
    uint16_t extraLen          = 0;
};
struct ZipDataDescriptor {
    uint32_t signature         = 0x08074b50;
    uint32_t crc32             = 0;
    uint32_t compressedSize    = 0;
    uint32_t uncompressedSize  = 0;
};
struct ZipCentralDirHeader {
    uint32_t signature         = 0x02014b50;
    uint16_t versionMade       = 20;
    uint16_t versionNeeded     = 20;
    uint16_t flags             = 0x0808;
    uint16_t method            = 0;
    uint16_t modTime           = 0;
    uint16_t modDate           = 0;
    uint32_t crc32             = 0;
    uint32_t compressedSize    = 0;
    uint32_t uncompressedSize  = 0;
    uint16_t nameLen           = 0;
    uint16_t extraLen          = 0;
    uint16_t commentLen        = 0;
    uint16_t diskNo            = 0;
    uint16_t internalAttr      = 0;
    uint32_t externalAttr      = 0;
    uint32_t localHeaderOffset = 0;
};
struct ZipEndOfCentralDir {
    uint32_t signature         = 0x06054b50;
    uint16_t diskNo            = 0;
    uint16_t centralDirDiskNo  = 0;
    uint16_t entriesThisDisk   = 0;
    uint16_t entriesTotal      = 0;
    uint32_t centralDirSize    = 0;
    uint32_t centralDirOffset  = 0;
    uint16_t commentLen        = 0;
};
#pragma pack(pop)

struct ZipEntryInfo {
    uint32_t compressedSize    = 0;
    uint32_t uncompressedSize  = 0;
    uint32_t localHeaderOffset = 0;
    uint32_t dataOffset        = 0;
};

struct ZipEntryView {
    const uint8_t* data = nullptr;
    size_t         size = 0;
};

struct ZipEntryRecord {
    std::string filename;
    uint32_t    crc32              = 0;
    uint32_t    size               = 0;
    uint32_t    localHeaderOffset  = 0;
};


// ----------------------------------------------------------------------
// ZipWriter: write a store-only ZIP archive
// ----------------------------------------------------------------------
class ZipWriter {
public:
    explicit ZipWriter(const std::string& zipPath)
        : out_(zipPath, std::ios::binary), path_(zipPath) {
        if (!out_.is_open())
            throw std::runtime_error("Cannot open zip file for writing: " + zipPath);
    }

    void writeStored(const std::string& filename, const uint8_t* data, size_t size) {
        if (size > std::numeric_limits<uint32_t>::max())
            throw std::runtime_error("Zip entry too large: " + filename);

        ZipEntryRecord entry;
        entry.filename          = filename;
        entry.crc32             = crc32Compute(data, size);
        entry.size              = static_cast<uint32_t>(size);
        entry.localHeaderOffset = static_cast<uint32_t>(out_.tellp());

        ZipLocalHeader local;
        local.nameLen = static_cast<uint16_t>(filename.size());
        writeRaw(local);
        writeBytes(reinterpret_cast<const uint8_t*>(filename.data()), filename.size());
        writeBytes(data, size);

        ZipDataDescriptor desc;
        desc.crc32            = entry.crc32;
        desc.compressedSize   = entry.size;
        desc.uncompressedSize = entry.size;
        writeRaw(desc);

        entries_.push_back(entry);
    }

    void close() {
        if (closed_) return;
        const uint32_t cdOffset = static_cast<uint32_t>(out_.tellp());
        for (const auto& e : entries_) {
            ZipCentralDirHeader cdir;
            cdir.crc32             = e.crc32;
            cdir.compressedSize    = e.size;
            cdir.uncompressedSize  = e.size;
            cdir.nameLen           = static_cast<uint16_t>(e.filename.size());
            cdir.localHeaderOffset = e.localHeaderOffset;
            writeRaw(cdir);
            writeBytes(reinterpret_cast<const uint8_t*>(e.filename.data()), e.filename.size());
        }
        const uint32_t cdEnd = static_cast<uint32_t>(out_.tellp());
        ZipEndOfCentralDir eocd;
        eocd.entriesThisDisk  = static_cast<uint16_t>(entries_.size());
        eocd.entriesTotal     = static_cast<uint16_t>(entries_.size());
        eocd.centralDirOffset = cdOffset;
        eocd.centralDirSize   = cdEnd - cdOffset;
        writeRaw(eocd);
        out_.close();
        closed_ = true;
    }

    ~ZipWriter() { try { close(); } catch (...) {} }

private:
    template <typename T>
    void writeRaw(const T& v) {
        out_.write(reinterpret_cast<const char*>(&v), sizeof(T));
        if (!out_.good())
            throw std::runtime_error("Failed writing zip file: " + path_);
    }
    void writeBytes(const uint8_t* data, size_t size) {
        if (!data || size == 0) return;
        out_.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(size));
        if (!out_.good())
            throw std::runtime_error("Failed writing zip payload: " + path_);
    }

    std::ofstream              out_;
    std::string                path_;
    std::vector<ZipEntryRecord> entries_;
    bool                       closed_ = false;
};


// ----------------------------------------------------------------------
// ZipArchiveReader: memory-resident zero-copy ZIP reader
// ----------------------------------------------------------------------
class ZipArchiveReader {
public:
    explicit ZipArchiveReader(const std::string& zipPath) : path_(zipPath) {
        std::ifstream in(zipPath, std::ios::binary | std::ios::ate);
        if (!in.is_open())
            throw std::runtime_error("Cannot open SOG zip file: " + zipPath);
        const std::streamoff fileSize = in.tellg();
        if (fileSize < static_cast<std::streamoff>(sizeof(ZipEndOfCentralDir)))
            throw std::runtime_error("SOG zip is too small: " + path_);
        fileData_.resize(static_cast<size_t>(fileSize));
        in.seekg(0, std::ios::beg);
        in.read(reinterpret_cast<char*>(fileData_.data()),
                static_cast<std::streamsize>(fileData_.size()));
        if (!in.good())
            throw std::runtime_error("Failed to read SOG zip bytes: " + path_);
        parseCentralDirectory();
    }

    bool hasEntry(const std::string& name) const {
        return entries_.count(name) != 0;
    }

    ZipEntryView getEntryView(const std::string& name) const {
        auto it = entries_.find(name);
        if (it == entries_.end())
            throw std::runtime_error("Missing SOG zip entry: " + name);
        const ZipEntryInfo& e = it->second;
        const size_t offset = static_cast<size_t>(e.dataOffset);
        const size_t size   = static_cast<size_t>(e.compressedSize);
        if (offset > fileData_.size() || size > fileData_.size() - offset)
            throw std::runtime_error("SOG zip entry range invalid: " + name);
        return ZipEntryView { fileData_.data() + offset, size };
    }

private:
    uint16_t readU16LE(const uint8_t* p) const {
        return static_cast<uint16_t>(p[0]) | (static_cast<uint16_t>(p[1]) << 8U);
    }
    uint32_t readU32LE(const uint8_t* p) const {
        return static_cast<uint32_t>(p[0]) |
               (static_cast<uint32_t>(p[1]) <<  8U) |
               (static_cast<uint32_t>(p[2]) << 16U) |
               (static_cast<uint32_t>(p[3]) << 24U);
    }

    void parseCentralDirectory() {
        const size_t fileSize  = fileData_.size();
        const size_t maxScan   = std::min<size_t>(fileSize, 65557);
        const size_t scanStart = fileSize - maxScan;

        size_t eocdPos = std::string::npos;
        for (size_t pos = fileSize - sizeof(ZipEndOfCentralDir); ; --pos) {
            if (readU32LE(fileData_.data() + pos) == 0x06054b50) {
                eocdPos = pos;
                break;
            }
            if (pos <= scanStart || pos == 0) break;
        }
        if (eocdPos == std::string::npos)
            throw std::runtime_error("Cannot find EOCD in SOG zip: " + path_);

        const uint8_t* eocd = fileData_.data() + eocdPos;
        const uint16_t total      = readU16LE(eocd + 10);
        const uint32_t cdSize     = readU32LE(eocd + 12);
        const uint32_t cdOffset   = readU32LE(eocd + 16);
        if (static_cast<size_t>(cdOffset) > fileSize ||
            static_cast<size_t>(cdSize)   > fileSize - static_cast<size_t>(cdOffset))
            throw std::runtime_error("SOG central directory range invalid: " + path_);

        size_t cursor = static_cast<size_t>(cdOffset);
        for (uint16_t i = 0; i < total; ++i) {
            if (cursor > fileSize || sizeof(ZipCentralDirHeader) > fileSize - cursor)
                throw std::runtime_error("Corrupted central directory in: " + path_);
            ZipCentralDirHeader cdir;
            std::memcpy(&cdir, fileData_.data() + cursor, sizeof(cdir));
            cursor += sizeof(cdir);
            if (cdir.signature != 0x02014b50)
                throw std::runtime_error("Invalid central dir header in: " + path_);
            if (cdir.method != 0)
                throw std::runtime_error("Unsupported compression in SOG zip: " + path_);

            const size_t metaSize = static_cast<size_t>(cdir.nameLen) +
                                    static_cast<size_t>(cdir.extraLen) +
                                    static_cast<size_t>(cdir.commentLen);
            if (cursor > fileSize || metaSize > fileSize - cursor)
                throw std::runtime_error("Corrupted central dir metadata in: " + path_);

            std::string filename(reinterpret_cast<const char*>(fileData_.data() + cursor),
                                 cdir.nameLen);
            cursor += metaSize;

            const size_t localOffset = static_cast<size_t>(cdir.localHeaderOffset);
            if (localOffset > fileSize || sizeof(ZipLocalHeader) > fileSize - localOffset)
                throw std::runtime_error("Corrupted local header offset in: " + path_);
            ZipLocalHeader local;
            std::memcpy(&local, fileData_.data() + localOffset, sizeof(local));
            if (local.signature != 0x04034b50)
                throw std::runtime_error("Invalid local header signature in: " + path_);
            if (local.method != 0)
                throw std::runtime_error("Only store method supported in SOG zip: " + path_);

            const size_t dataOffset = localOffset + sizeof(ZipLocalHeader) +
                                      static_cast<size_t>(local.nameLen) +
                                      static_cast<size_t>(local.extraLen);
            if (dataOffset > fileSize ||
                static_cast<size_t>(cdir.compressedSize) > fileSize - dataOffset)
                throw std::runtime_error("Corrupted SOG payload range in: " + path_);

            entries_[filename] = ZipEntryInfo {
                cdir.compressedSize,
                cdir.uncompressedSize,
                cdir.localHeaderOffset,
                static_cast<uint32_t>(dataOffset)
            };
        }
    }

    std::string                                  path_;
    std::vector<uint8_t>                         fileData_;
    std::unordered_map<std::string, ZipEntryInfo> entries_;
};


// ----------------------------------------------------------------------
// Dynamic libwebp loader (runtime dlopen to keep optional)
// ----------------------------------------------------------------------
struct DecodedWebPImage {
    int                  width  = 0;
    int                  height = 0;
    std::vector<uint8_t> rgba;
};

using WebPEncodeLosslessRGBAFn = size_t   (*)(const uint8_t*, int, int, int, uint8_t**);
using WebPDecodeRGBAFn         = uint8_t* (*)(const uint8_t*, size_t, int*, int*);
using WebPGetInfoFn            = int      (*)(const uint8_t*, size_t, int*, int*);
using WebPDecodeRGBAIntoFn     = uint8_t* (*)(const uint8_t*, size_t, uint8_t*, size_t, int);
using WebPFreeFn               = void     (*)(void*);

class DynamicWebP {
public:
    static DynamicWebP& instance() {
        static DynamicWebP lib;
        return lib;
    }

    bool canEncode() const {
        return handle_ && encodeLosslessRGBAFnPtr_ && freeFn_;
    }
    bool canDecode() const {
        return handle_ && (
            (decodeRGBAIntoFnPtr_ && getInfoFnPtr_) ||
            (decodeRGBAFnPtr_     && freeFn_));
    }

    std::vector<uint8_t> encodeLosslessRGBA(
        const std::vector<uint8_t>& rgba, int w, int h) const
    {
        if (!canEncode())
            throw std::runtime_error("libwebp encoder unavailable for SOG export.");
        uint8_t* buf = nullptr;
        const size_t sz = encodeLosslessRGBAFnPtr_(rgba.data(), w, h, w * 4, &buf);
        if (!sz || !buf)
            throw std::runtime_error("WebP lossless encoding failed.");
        std::vector<uint8_t> out(buf, buf + sz);
        freeFn_(buf);
        return out;
    }

    DecodedWebPImage decodeRGBA(const uint8_t* data, size_t size) const {
        if (!canDecode())
            throw std::runtime_error("libwebp decoder unavailable for SOG import.");
        if (!data || !size)
            throw std::runtime_error("WebP decode input is empty.");

        DecodedWebPImage result;
        if (decodeRGBAIntoFnPtr_ && getInfoFnPtr_) {
            if (!getInfoFnPtr_(data, size, &result.width, &result.height) ||
                result.width <= 0 || result.height <= 0)
                throw std::runtime_error("WebP info decode failed.");
            const size_t bytes = static_cast<size_t>(result.width) *
                                 static_cast<size_t>(result.height) * 4;
            result.rgba.resize(bytes);
            if (!decodeRGBAIntoFnPtr_(data, size, result.rgba.data(), bytes,
                                      result.width * 4))
                throw std::runtime_error("WebP decode-into failed.");
            return result;
        }
        uint8_t* raw = decodeRGBAFnPtr_(data, size, &result.width, &result.height);
        if (!raw || result.width <= 0 || result.height <= 0)
            throw std::runtime_error("WebP decode failed.");
        const size_t bytes = static_cast<size_t>(result.width) *
                             static_cast<size_t>(result.height) * 4;
        result.rgba.assign(raw, raw + bytes);
        freeFn_(raw);
        return result;
    }

private:
#if defined(_WIN32)
    static void* dynOpen(const char* n)               { return (void*)LoadLibraryA(n); }
    static void* dynSym(void* h, const char* s)       { return (void*)GetProcAddress((HMODULE)h, s); }
    static void  dynClose(void* h)                    { FreeLibrary((HMODULE)h); }
#else
    static void* dynOpen(const char* n)               { return dlopen(n, RTLD_LAZY); }
    static void* dynSym(void* h, const char* s)       { return dlsym(h, s); }
    static void  dynClose(void* h)                    { dlclose(h); }
#endif

    DynamicWebP() {
#if defined(_WIN32)
        handle_ = dynOpen("libwebp.dll");
        if (!handle_) handle_ = dynOpen("webp.dll");
#else
        handle_ = dynOpen("libwebp.so.7");
        if (!handle_) handle_ = dynOpen("libwebp.so");
#endif
        if (!handle_) return;

        encodeLosslessRGBAFnPtr_ = reinterpret_cast<WebPEncodeLosslessRGBAFn>(
            dynSym(handle_, "WebPEncodeLosslessRGBA"));
        decodeRGBAFnPtr_     = reinterpret_cast<WebPDecodeRGBAFn>    (dynSym(handle_, "WebPDecodeRGBA"));
        getInfoFnPtr_        = reinterpret_cast<WebPGetInfoFn>        (dynSym(handle_, "WebPGetInfo"));
        decodeRGBAIntoFnPtr_ = reinterpret_cast<WebPDecodeRGBAIntoFn>(dynSym(handle_, "WebPDecodeRGBAInto"));
        freeFn_              = reinterpret_cast<WebPFreeFn>           (dynSym(handle_, "WebPFree"));

        const bool hasDecoder = (decodeRGBAIntoFnPtr_ && getInfoFnPtr_) ||
                                (decodeRGBAFnPtr_ && freeFn_);
        const bool hasEncoder = encodeLosslessRGBAFnPtr_ && freeFn_;
        if (!hasDecoder && !hasEncoder) {
            dynClose(handle_);
            handle_                  = nullptr;
            encodeLosslessRGBAFnPtr_ = nullptr;
            decodeRGBAFnPtr_         = nullptr;
            getInfoFnPtr_            = nullptr;
            decodeRGBAIntoFnPtr_     = nullptr;
            freeFn_                  = nullptr;
        }
    }
    ~DynamicWebP() { if (handle_) dynClose(handle_); }

    void* handle_                          = nullptr;
    WebPEncodeLosslessRGBAFn encodeLosslessRGBAFnPtr_ = nullptr;
    WebPDecodeRGBAFn         decodeRGBAFnPtr_         = nullptr;
    WebPGetInfoFn            getInfoFnPtr_            = nullptr;
    WebPDecodeRGBAIntoFn     decodeRGBAIntoFnPtr_     = nullptr;
    WebPFreeFn               freeFn_                  = nullptr;
};


// ----------------------------------------------------------------------
// SOG quantization helpers
// ----------------------------------------------------------------------
inline float logTransform(float v) {
    return std::copysign(std::log1p(std::fabs(v)), v);
}
inline float invLogTransform(float v) {
    return std::copysign(std::expm1(std::fabs(v)), v);
}
inline uint16_t quantizeToU16(float v, float mn, float mx) {
    const float range = mx - mn;
    if (range <= kQuantEpsilon) return 0;
    return static_cast<uint16_t>(std::lround(
        std::clamp((v - mn) / range, 0.f, 1.f) * 65535.f));
}
inline float dequantizeFromU16(uint16_t q, float mn, float mx) {
    const float range = mx - mn;
    if (range <= kQuantEpsilon) return mn;
    return mn + (static_cast<float>(q) / 65535.f) * range;
}
inline uint8_t quantizeToU8(float v, float mn, float mx) {
    const float range = mx - mn;
    if (range <= kQuantEpsilon) return 0;
    return static_cast<uint8_t>(std::lround(
        std::clamp((v - mn) / range, 0.f, 1.f) * 255.f));
}

template <typename ValueGetter>
void buildUniformCodebookAndLabels(
    size_t                                    count,
    ValueGetter                               getter,
    std::array<float, kSogCodebookSize>&      codebook,
    std::vector<uint8_t>&                     labels)
{
    labels.assign(count, 0);
    if (!count) { codebook.fill(0.f); return; }

    float mn = std::numeric_limits<float>::infinity();
    float mx = -std::numeric_limits<float>::infinity();
    for (size_t i = 0; i < count; ++i) {
        const float v = getter(i);
        mn = std::min(mn, v);
        mx = std::max(mx, v);
    }
    if (!std::isfinite(mn) || !std::isfinite(mx))
        throw std::runtime_error("SOG quantization: non-finite input values.");

    const float range = mx - mn;
    if (range <= kQuantEpsilon) { codebook.fill(mn); return; }

    for (size_t i = 0; i < kSogCodebookSize; ++i)
        codebook[i] = mn + static_cast<float>(i) /
                      static_cast<float>(kSogCodebookSize - 1) * range;

    const float invRange = static_cast<float>(kSogCodebookSize - 1) / range;
    for (size_t i = 0; i < count; ++i) {
        const int q = static_cast<int>(std::lround((getter(i) - mn) * invRange));
        labels[i] = static_cast<uint8_t>(
            std::clamp(q, 0, static_cast<int>(kSogCodebookSize - 1)));
    }
}


// ----------------------------------------------------------------------
// Quaternion packing / unpacking
// ----------------------------------------------------------------------
std::array<uint8_t, 4> packQuaternion(const Rot& qIn) {
    Rot q = qIn;
    if (!std::isfinite(q[0]) || !std::isfinite(q[1]) ||
        !std::isfinite(q[2]) || !std::isfinite(q[3]) ||
        q.norm() <= kQuantEpsilon)
        q = Rot(1.f, 0.f, 0.f, 0.f);
    q.normalize();

    std::array<float, 4> f = {q[0], q[1], q[2], q[3]};
    int maxComp = 0;
    for (int i = 1; i < 4; ++i)
        if (std::fabs(f[i]) > std::fabs(f[maxComp])) maxComp = i;
    if (f[maxComp] < 0.f)
        for (float& c : f) c = -c;

    constexpr std::array<std::array<int,3>, 4> kLut = {{
        {{1,2,3}}, {{0,2,3}}, {{0,1,3}}, {{0,1,2}}
    }};
    const float sq2 = std::sqrt(2.f);
    for (float& c : f) c *= sq2;
    const auto& idx = kLut[maxComp];
    return {
        quantizeToU8(f[idx[0]] * .5f + .5f, 0.f, 1.f),
        quantizeToU8(f[idx[1]] * .5f + .5f, 0.f, 1.f),
        quantizeToU8(f[idx[2]] * .5f + .5f, 0.f, 1.f),
        static_cast<uint8_t>(kQuatTagBase + maxComp)
    };
}

Rot unpackQuaternion(const uint8_t packed[4]) {
    const uint8_t tag = packed[3];
    if (tag < kQuatTagBase || tag > kQuatTagBase + 3)
        return Rot(1.f, 0.f, 0.f, 0.f);
    const int maxComp = static_cast<int>(tag - kQuatTagBase);
    constexpr std::array<std::array<int,3>, 4> kLut = {{
        {{1,2,3}}, {{0,2,3}}, {{0,1,3}}, {{0,1,2}}
    }};
    const float sq2 = std::sqrt(2.f);
    const float a = (packed[0] / 255.f) * 2.f - 1.f;
    const float b = (packed[1] / 255.f) * 2.f - 1.f;
    const float c = (packed[2] / 255.f) * 2.f - 1.f;
    std::array<float,4> q = {0.f, 0.f, 0.f, 0.f};
    const auto& idx = kLut[maxComp];
    q[idx[0]] = a / sq2;
    q[idx[1]] = b / sq2;
    q[idx[2]] = c / sq2;
    const float sumSq = q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3];
    q[maxComp] = std::sqrt(std::max(0.f, 1.f - sumSq));
    Rot result(q[0], q[1], q[2], q[3]);
    if (result.norm() <= kQuantEpsilon) return Rot(1.f, 0.f, 0.f, 0.f);
    return result.normalized();
}


// ----------------------------------------------------------------------
// SOG SH degree inference
// ----------------------------------------------------------------------
int inferShDegreeFromMeta(const nlohmann::json& meta) {
    if (meta.contains("shN") && meta["shN"].is_object()) {
        const auto& shN = meta["shN"];
        if (shN.contains("bands") && shN["bands"].is_number_integer())
            return std::clamp(shN["bands"].get<int>(), 0, 3);
    }
    return 0;
}


// ----------------------------------------------------------------------
// exportSupersplatSogBundle: write a scene to the SOG format.
// Templated on D so it can access the typed SHs<D> arrays.
// ----------------------------------------------------------------------
template <int D>
void exportSupersplatSogBundle(const LoadResult<D>& scene,
                               const std::string&   sogPath)
{
    DynamicWebP& webp = DynamicWebP::instance();
    if (!webp.canEncode())
        throw std::runtime_error("libwebp encoder unavailable for SOG export.");

    const int numRows = scene.numPoints;
    const int width  = std::max(4, static_cast<int>(
        std::ceil(std::sqrt(static_cast<double>(std::max(1, numRows))) / 4.0) * 4.0));
    const int height = std::max(4, static_cast<int>(
        std::ceil(static_cast<double>(std::max(1, numRows)) /
                  static_cast<double>(width) / 4.0) * 4.0));
    const size_t pixelCount = static_cast<size_t>(width) * static_cast<size_t>(height);
    const size_t texelSize  = pixelCount * 4;

    std::vector<uint8_t> meansL(texelSize, 0), meansU(texelSize, 0),
                         quats (texelSize, 0), scales(texelSize, 0),
                         sh0   (texelSize, 0);

    std::array<float,3> meansMin = {0.f,0.f,0.f}, meansMax = {0.f,0.f,0.f};
    if (numRows > 0) {
        for (int c = 0; c < 3; ++c) {
            meansMin[c] =  std::numeric_limits<float>::infinity();
            meansMax[c] = -std::numeric_limits<float>::infinity();
        }
        for (int i = 0; i < numRows; ++i)
            for (int c = 0; c < 3; ++c) {
                const float v = logTransform(scene.points[i][c]);
                meansMin[c] = std::min(meansMin[c], v);
                meansMax[c] = std::max(meansMax[c], v);
            }
    }

    std::array<float, kSogCodebookSize> scaleCodebook, colorCodebook;
    std::vector<uint8_t> scaleLabels, colorLabels;

    buildUniformCodebookAndLabels(static_cast<size_t>(numRows) * 3,
        [&](size_t idx) {
            return std::log(std::max(scene.scale[idx/3][idx%3], 1e-12f));
        }, scaleCodebook, scaleLabels);

    buildUniformCodebookAndLabels(static_cast<size_t>(numRows) * 3,
        [&](size_t idx) { return scene.shs[idx/3][idx%3]; },
        colorCodebook, colorLabels);

    const int shBands = D;
    const int kBandToCoeff[4] = {0, 3, 8, 15};
    const int shCoeffs    = kBandToCoeff[shBands];
    const int shDimension = shCoeffs * 3;
    const bool hasShN = (shBands > 0 && shCoeffs > 0 && numRows > 0);

    int shPaletteCount = 0;
    std::array<float, kSogCodebookSize> shNCodebook{};
    std::vector<uint16_t> shNLabels16;
    std::vector<uint8_t>  shNCentroidsWebp, shNLabelsWebp;

    if (hasShN) {
        if (numRows < 1024) {
            int p2 = 1;
            while ((p2 << 1) <= numRows && p2 < 65536) p2 <<= 1;
            shPaletteCount = p2;
        } else {
            int factor = numRows / 1024;
            int p2 = 1;
            while ((p2 << 1) <= factor && p2 < 64) p2 <<= 1;
            shPaletteCount = p2 * 1024;
        }
        shPaletteCount = std::clamp(shPaletteCount, 1, std::min(numRows, 65536));

        const int groupSize = std::max(1, (numRows + shPaletteCount - 1) / shPaletteCount);
        shNLabels16.resize(numRows, 0);
        std::vector<float> centSums(static_cast<size_t>(shPaletteCount) *
                                    static_cast<size_t>(shDimension), 0.f);
        std::vector<int>   centCounts(shPaletteCount, 0);

        for (int i = 0; i < numRows; ++i) {
            const int label = std::min(i / groupSize, shPaletteCount - 1);
            shNLabels16[i] = static_cast<uint16_t>(label);
            centCounts[label]++;
            for (int coeff = 0; coeff < shCoeffs; ++coeff) {
                const int shBase = (coeff + 1) * 3;
                const size_t sb  = static_cast<size_t>(label) * shDimension;
                centSums[sb + coeff]              += scene.shs[i][shBase + 0];
                centSums[sb + shCoeffs + coeff]   += scene.shs[i][shBase + 1];
                centSums[sb + shCoeffs*2 + coeff] += scene.shs[i][shBase + 2];
            }
        }
        for (int label = 0; label < shPaletteCount; ++label) {
            if (centCounts[label] <= 0) continue;
            const float inv = 1.f / static_cast<float>(centCounts[label]);
            const size_t sb = static_cast<size_t>(label) * shDimension;
            for (int d = 0; d < shDimension; ++d) centSums[sb + d] *= inv;
        }

        std::vector<uint8_t> shNCentroidLabels;
        buildUniformCodebookAndLabels(
            static_cast<size_t>(shPaletteCount) * shDimension,
            [&](size_t idx) { return centSums[idx]; },
            shNCodebook, shNCentroidLabels);

        const int shCW = 64 * shCoeffs;
        const int shCH = std::max(1, (shPaletteCount + 63) / 64);
        std::vector<uint8_t> shNCentroids(
            static_cast<size_t>(shCW) * static_cast<size_t>(shCH) * 4, 0);

        for (int label = 0; label < shPaletteCount; ++label) {
            const int cxBase = (label % 64) * shCoeffs;
            const int cy     = label / 64;
            for (int coeff = 0; coeff < shCoeffs; ++coeff) {
                const size_t srcBase   = static_cast<size_t>(label) * shDimension;
                const size_t pixelBase = (static_cast<size_t>(cy) * shCW +
                                          static_cast<size_t>(cxBase + coeff)) * 4;
                shNCentroids[pixelBase + 0] = shNCentroidLabels[srcBase + coeff];
                shNCentroids[pixelBase + 1] = shNCentroidLabels[srcBase + shCoeffs + coeff];
                shNCentroids[pixelBase + 2] = shNCentroidLabels[srcBase + shCoeffs*2 + coeff];
                shNCentroids[pixelBase + 3] = 0xFF;
            }
        }

        std::vector<uint8_t> shNLabelsTexture(texelSize, 0);
        for (int i = 0; i < numRows; ++i) {
            const size_t base = static_cast<size_t>(i) * 4;
            const uint16_t lbl = shNLabels16[i];
            shNLabelsTexture[base + 0] = static_cast<uint8_t>(lbl & 0xFFu);
            shNLabelsTexture[base + 1] = static_cast<uint8_t>((lbl >> 8u) & 0xFFu);
            shNLabelsTexture[base + 2] = 0;
            shNLabelsTexture[base + 3] = 0xFF;
        }
        shNCentroidsWebp = webp.encodeLosslessRGBA(shNCentroids, shCW, shCH);
        shNLabelsWebp    = webp.encodeLosslessRGBA(shNLabelsTexture, width, height);
    }

    for (int i = 0; i < numRows; ++i) {
        const size_t base = static_cast<size_t>(i) * 4;
        for (int c = 0; c < 3; ++c) {
            const uint16_t enc = quantizeToU16(
                logTransform(scene.points[i][c]), meansMin[c], meansMax[c]);
            meansL[base + c] = static_cast<uint8_t>(enc & 0xFFu);
            meansU[base + c] = static_cast<uint8_t>((enc >> 8u) & 0xFFu);
        }
        meansL[base + 3] = 0xFF;
        meansU[base + 3] = 0xFF;

        const auto pq = packQuaternion(scene.rot[i]);
        quats[base + 0] = pq[0]; quats[base + 1] = pq[1];
        quats[base + 2] = pq[2]; quats[base + 3] = pq[3];

        const size_t sb = static_cast<size_t>(i) * 3;
        scales[base + 0] = scaleLabels[sb + 0];
        scales[base + 1] = scaleLabels[sb + 1];
        scales[base + 2] = scaleLabels[sb + 2];
        scales[base + 3] = 0xFF;

        sh0[base + 0] = colorLabels[sb + 0];
        sh0[base + 1] = colorLabels[sb + 1];
        sh0[base + 2] = colorLabels[sb + 2];
        sh0[base + 3] = static_cast<uint8_t>(
            std::clamp(std::lround(scene.opacity[i] * 255.f), 0L, 255L));
    }

    auto meansLWebp = webp.encodeLosslessRGBA(meansL, width, height);
    auto meansUWebp = webp.encodeLosslessRGBA(meansU, width, height);
    auto quatsWebp  = webp.encodeLosslessRGBA(quats,  width, height);
    auto scalesWebp = webp.encodeLosslessRGBA(scales, width, height);
    auto sh0Webp    = webp.encodeLosslessRGBA(sh0,    width, height);

    std::ostringstream meta;
    meta << std::setprecision(9);
    meta << "{\"version\":2,\"asset\":{\"generator\":\"OptiSplat\"},\"count\":" << numRows;
    meta << ",\"means\":{\"mins\":["
         << meansMin[0] << "," << meansMin[1] << "," << meansMin[2] << "]"
         << ",\"maxs\":["
         << meansMax[0] << "," << meansMax[1] << "," << meansMax[2] << "]"
         << ",\"files\":[\"means_l.webp\",\"means_u.webp\"]}";
    meta << ",\"scales\":{\"codebook\":[";
    for (size_t i = 0; i < scaleCodebook.size(); ++i) {
        if (i) meta << ",";
        meta << scaleCodebook[i];
    }
    meta << "],\"files\":[\"scales.webp\"]}";
    meta << ",\"quats\":{\"files\":[\"quats.webp\"]}";
    meta << ",\"sh0\":{\"codebook\":[";
    for (size_t i = 0; i < colorCodebook.size(); ++i) {
        if (i) meta << ",";
        meta << colorCodebook[i];
    }
    meta << "],\"files\":[\"sh0.webp\"]}";
    if (hasShN) {
        meta << ",\"shN\":{\"count\":" << shPaletteCount
             << ",\"bands\":" << shBands
             << ",\"codebook\":[";
        for (size_t i = 0; i < shNCodebook.size(); ++i) {
            if (i) meta << ",";
            meta << shNCodebook[i];
        }
        meta << "],\"files\":[\"shN_centroids.webp\",\"shN_labels.webp\"]}";
    }
    meta << "}";
    const std::string metaText = meta.str();

    ZipWriter zip(sogPath);
    zip.writeStored("means_l.webp", meansLWebp.data(), meansLWebp.size());
    zip.writeStored("means_u.webp", meansUWebp.data(), meansUWebp.size());
    zip.writeStored("quats.webp",   quatsWebp.data(),  quatsWebp.size());
    zip.writeStored("scales.webp",  scalesWebp.data(), scalesWebp.size());
    zip.writeStored("sh0.webp",     sh0Webp.data(),    sh0Webp.size());
    if (hasShN) {
        zip.writeStored("shN_centroids.webp", shNCentroidsWebp.data(), shNCentroidsWebp.size());
        zip.writeStored("shN_labels.webp",    shNLabelsWebp.data(),    shNLabelsWebp.size());
    }
    zip.writeStored("meta.json",
        reinterpret_cast<const uint8_t*>(metaText.data()), metaText.size());
    zip.close();

    GS_INFO("SOG cache written to %s", sogPath.c_str());
}

} // anonymous namespace


// ======================================================================
// PlyLoader implementation
// ======================================================================

template <int D>
bool PlyLoader<D>::peekInfo(const std::string& path,
                            int*               outShDegree,
                            int*               outNumPoints)
{
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) return false;

    std::string line;
    bool  headerEnd    = false;
    int   numVertex    = 0;
    int   numCoeff     = 0;

    while (std::getline(file, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line == "end_header") { headerEnd = true; break; }
        std::istringstream iss(line);
        std::string kw;
        iss >> kw;
        if (kw == "element") {
            std::string name;
            iss >> name;
            if (name == "vertex") iss >> numVertex;
        } else if (kw == "property") {
            std::string ty, name;
            iss >> ty >> name;
            if (name.find("f_dc") == 0 || name.find("f_rest") == 0)
                numCoeff++;
        }
    }
    if (!headerEnd || numVertex <= 0 || numCoeff % 3 != 0) return false;
    if (outNumPoints)  *outNumPoints  = numVertex;
    if (outShDegree)   *outShDegree   = static_cast<int>(std::sqrt(numCoeff / 3)) - 1;
    return true;
}

template <int D>
void PlyLoader<D>::load(const std::string& path, LoadResult<D>& result) {
    std::ifstream infile(path, std::ios_base::binary);
    if (!infile.good())
        throw std::runtime_error("Cannot open PLY file: " + path);

    // --- 1) Parse header ---
    int numPoints = -1;
    std::vector<std::string> props;
    int totalFloats = 0;
    std::string line;
    while (std::getline(infile, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        std::istringstream iss(line);
        std::string token;
        iss >> token;
        if (token == "element") {
            std::string type;
            iss >> type;
            if (type == "vertex") iss >> numPoints;
        } else if (token == "property") {
            std::string ty, name;
            iss >> ty >> name;
            props.push_back(name);
            totalFloats++;
        } else if (token == "end_header") {
            break;
        }
    }
    if (numPoints < 0 || totalFloats < 0)
        throw std::runtime_error("PLY parse failed (no vertex count): " + path);

    result.numPoints = numPoints;
    result.points.resize(numPoints);
    result.shs.resize(numPoints);
    result.opacity.resize(numPoints);
    result.scale.resize(numPoints);
    result.rot.resize(numPoints);

    // --- 2) Map property names to channels ---
    struct PropInfo {
        enum Type { POS, F_DC, F_REST, OPACITY, SCALE, ROT, UNKNOWN };
        Type type    = UNKNOWN;
        int  channel = -1;
    };
    const int SH_N = (D + 1) * (D + 1);
    std::vector<PropInfo> propInfos(props.size());
    for (size_t j = 0; j < props.size(); ++j) {
        const std::string& n = props[j];
        if      (n == "x") propInfos[j] = {PropInfo::POS, 0};
        else if (n == "y") propInfos[j] = {PropInfo::POS, 1};
        else if (n == "z") propInfos[j] = {PropInfo::POS, 2};
        else if (n.rfind("f_dc_", 0) == 0)
            propInfos[j] = {PropInfo::F_DC, std::stoi(n.substr(5))};
        else if (n.rfind("f_rest_", 0) == 0) {
            int num = std::stoi(n.substr(7));
            int ch  = num / (SH_N - 1);
            int idx = num % (SH_N - 1) + 1;
            propInfos[j] = {PropInfo::F_REST, idx * 3 + ch};
        }
        else if (n == "opacity")                propInfos[j] = {PropInfo::OPACITY, 0};
        else if (n.rfind("scale_", 0) == 0)     propInfos[j] = {PropInfo::SCALE, std::stoi(n.substr(6))};
        else if (n.rfind("rot_", 0) == 0)       propInfos[j] = {PropInfo::ROT,   std::stoi(n.substr(4))};
    }

    // --- 3) Read binary data ---
    std::vector<float> buffer(static_cast<size_t>(totalFloats) *
                              static_cast<size_t>(numPoints));
    infile.read(reinterpret_cast<char*>(buffer.data()),
                static_cast<std::streamsize>(buffer.size() * sizeof(float)));

    Eigen::Vector3f minPos( FLT_MAX,  FLT_MAX,  FLT_MAX);
    Eigen::Vector3f maxPos(-FLT_MAX, -FLT_MAX, -FLT_MAX);

    for (int i = 0; i < numPoints; ++i) {
        const size_t base = static_cast<size_t>(i) * totalFloats;
        for (size_t j = 0; j < props.size(); ++j) {
            const float v = buffer[base + j];
            const PropInfo& pi = propInfos[j];
            switch (pi.type) {
                case PropInfo::POS:     result.points[i][pi.channel] = v;                break;
                case PropInfo::F_DC:    result.shs[i][pi.channel]    = v;                break;
                case PropInfo::F_REST:  result.shs[i][pi.channel]    = v;                break;
                case PropInfo::OPACITY: result.opacity[i]             = sigmoid(v);       break;
                case PropInfo::SCALE:   result.scale[i][pi.channel]   = std::exp(v);     break;
                case PropInfo::ROT:     result.rot[i][pi.channel]     = v;               break;
                default: break;
            }
        }
        result.rot[i] = result.rot[i].normalized();

        const Eigen::Vector3f& p = result.points[i];
        minPos = minPos.cwiseMin(p);
        maxPos = maxPos.cwiseMax(p);
    }

    result.sceneMin = minPos;
    result.sceneMax = maxPos;

    GS_INFO("PlyLoader: loaded %d gaussians from %s", numPoints, path.c_str());

    // --- 4) Write SOG cache if requested ---
    if (!cachePath_.empty()) {
        const std::filesystem::path cacheDir =
            std::filesystem::path(cachePath_).parent_path();
        if (!cacheDir.empty() && !std::filesystem::exists(cacheDir)) {
            std::filesystem::create_directories(cacheDir);
            GS_INFO("SOG cache directory created: %s", cacheDir.string().c_str());
        }
        auto t0 = Utils::nowUs();
        exportSupersplatSogBundle<D>(result, cachePath_);
        GS_INFO("SOG cache saved to %s (%.2f ms)",
                cachePath_.c_str(),
                static_cast<float>(Utils::nowUs() - t0) / 1000.f);
    }
}


// ======================================================================
// SogLoader implementation
// ======================================================================

template <int D>
bool SogLoader<D>::peekInfo(const std::string& path,
                            int*               outShDegree,
                            int*               outNumPoints)
{
    try {
        ZipArchiveReader zip(path);
        if (!zip.hasEntry("meta.json")) return false;
        const ZipEntryView v = zip.getEntryView("meta.json");
        const nlohmann::json meta = nlohmann::json::parse(v.data, v.data + v.size);
        if (!meta.contains("count") || !meta["count"].is_number_integer()) return false;
        if (outNumPoints) *outNumPoints = meta["count"].get<int>();
        if (outShDegree)  *outShDegree  = inferShDegreeFromMeta(meta);
        return true;
    } catch (...) {
        return false;
    }
}

template <int D>
void SogLoader<D>::load(const std::string& path, LoadResult<D>& result) {
    if (!std::filesystem::exists(path))
        throw std::runtime_error("SOG file does not exist: " + path);

    GS_INFO("SogLoader: loading from %s", path.c_str());
    auto t0            = Utils::nowUs();
    auto tMetaDone     = t0;
    auto tDecodeDone   = t0;
    auto tShNPrepDone  = t0;
    auto tPointLoopDone = t0;

    DynamicWebP& webp = DynamicWebP::instance();
    if (!webp.canDecode())
        throw std::runtime_error("libwebp decoder unavailable for SOG import.");

    ZipArchiveReader zip(path);
    if (!zip.hasEntry("meta.json"))
        throw std::runtime_error("SOG missing meta.json: " + path);

    const ZipEntryView metaView = zip.getEntryView("meta.json");
    const nlohmann::json meta = nlohmann::json::parse(
        metaView.data, metaView.data + metaView.size);
    tMetaDone = Utils::nowUs();

    if (!meta.contains("count") || !meta["count"].is_number_integer())
        throw std::runtime_error("Invalid SOG meta count: " + path);

    const int fileShDegree = inferShDegreeFromMeta(meta);
    if (fileShDegree > D)
        throw std::runtime_error(
            "SOG SH degree mismatch: file=" + std::to_string(fileShDegree) +
            " renderer=" + std::to_string(D));
    if (fileShDegree < D)
        GS_WARNING("SOG SH degree (%d) < renderer degree (%d), missing bands zero-filled.",
                   fileShDegree, D);

    const int numPoints = meta["count"].get<int>();
    if (numPoints < 0) throw std::runtime_error("Invalid SOG point count.");

    result.numPoints = numPoints;
    result.points.resize(numPoints);
    result.shs.resize(numPoints);
    result.opacity.resize(numPoints);
    result.scale.resize(numPoints);
    result.rot.resize(numPoints);

    // --- Decode metadata for texture entries ---
    const auto& meansObj   = meta.at("means");
    const auto& meansFiles = meansObj.at("files");
    const auto& quatsObj   = meta.at("quats");
    const auto& scalesObj  = meta.at("scales");
    const auto& sh0Obj     = meta.at("sh0");

    std::array<float, kSogCodebookSize> scaleCodebook, sh0Codebook;
    {
        const auto& cb = scalesObj.at("codebook");
        if (!cb.is_array() || cb.size() != kSogCodebookSize)
            throw std::runtime_error("Invalid SOG scales codebook.");
        for (size_t i = 0; i < kSogCodebookSize; ++i)
            scaleCodebook[i] = cb.at(i).get<float>();
    }
    {
        const auto& cb = sh0Obj.at("codebook");
        if (!cb.is_array() || cb.size() != kSogCodebookSize)
            throw std::runtime_error("Invalid SOG sh0 codebook.");
        for (size_t i = 0; i < kSogCodebookSize; ++i)
            sh0Codebook[i] = cb.at(i).get<float>();
    }

    // --- Async-decode base textures ---
    const ZipEntryView meansLView = zip.getEntryView(meansFiles.at(0).get<std::string>());
    const ZipEntryView meansUView = zip.getEntryView(meansFiles.at(1).get<std::string>());
    const ZipEntryView quatsView  = zip.getEntryView(quatsObj.at("files").at(0).get<std::string>());
    const ZipEntryView scalesView = zip.getEntryView(scalesObj.at("files").at(0).get<std::string>());
    const ZipEntryView sh0View    = zip.getEntryView(sh0Obj.at("files").at(0).get<std::string>());

    auto meansLTask  = std::async(std::launch::async, [&]{ return webp.decodeRGBA(meansLView.data, meansLView.size); });
    auto meansUTask  = std::async(std::launch::async, [&]{ return webp.decodeRGBA(meansUView.data, meansUView.size); });
    auto quatsTask   = std::async(std::launch::async, [&]{ return webp.decodeRGBA(quatsView.data,  quatsView.size); });
    auto scalesTask  = std::async(std::launch::async, [&]{ return webp.decodeRGBA(scalesView.data, scalesView.size); });
    auto sh0Task     = std::async(std::launch::async, [&]{ return webp.decodeRGBA(sh0View.data,    sh0View.size); });

    const DecodedWebPImage meansL = meansLTask.get();
    const DecodedWebPImage meansU = meansUTask.get();
    const DecodedWebPImage quats  = quatsTask.get();
    const DecodedWebPImage scales = scalesTask.get();
    const DecodedWebPImage sh0    = sh0Task.get();
    tDecodeDone = Utils::nowUs();

    // --- Basic sanity checks ---
    if (meansL.width != meansU.width || meansL.height != meansU.height)
        throw std::runtime_error("SOG means textures size mismatch.");
    auto checkSize = [&](const DecodedWebPImage& img, const char* name) {
        if (static_cast<size_t>(img.width) * static_cast<size_t>(img.height) <
            static_cast<size_t>(numPoints))
            throw std::runtime_error(std::string("SOG ") + name +
                                     " texture too small for point count.");
    };
    checkSize(meansL, "means"); checkSize(quats, "quats");
    checkSize(scales, "scales"); checkSize(sh0, "sh0");

    const auto& mins     = meansObj.at("mins");
    const auto& maxs     = meansObj.at("maxs");
    const std::array<float,3> meansMin = {mins.at(0).get<float>(), mins.at(1).get<float>(), mins.at(2).get<float>()};
    const std::array<float,3> meansMax = {maxs.at(0).get<float>(), maxs.at(1).get<float>(), maxs.at(2).get<float>()};

    // --- Optional shN ---
    const int shExtraPerPoint = ((D+1)*(D+1) - 1) * 3;
    bool hasShN = false;
    int shBands = 0, shCoeffs = 0, shNPaletteCount = 0;
    std::array<float, kSogCodebookSize> shNCodebook{};
    DecodedWebPImage shNCentroids, shNLabels;

    if (meta.contains("shN") && meta["shN"].is_object()) {
        const auto& shNObj = meta["shN"];
        if (shNObj.contains("bands") && shNObj["bands"].is_number_integer()) {
            shBands  = std::clamp(shNObj["bands"].get<int>(), 0, 3);
            const int kBandToCoeff[4] = {0, 3, 8, 15};
            shCoeffs = kBandToCoeff[shBands];
        }
        if (shCoeffs > 0 && shExtraPerPoint > 0) {
            const auto& cb = shNObj.at("codebook");
            if (!cb.is_array() || cb.size() != kSogCodebookSize)
                throw std::runtime_error("Invalid SOG shN codebook.");
            for (size_t i = 0; i < kSogCodebookSize; ++i)
                shNCodebook[i] = cb.at(i).get<float>();

            const auto& shNFiles = shNObj.at("files");
            const ZipEntryView centV = zip.getEntryView(shNFiles.at(0).get<std::string>());
            const ZipEntryView lblV  = zip.getEntryView(shNFiles.at(1).get<std::string>());
            auto centTask = std::async(std::launch::async, [&]{ return webp.decodeRGBA(centV.data, centV.size); });
            auto lblTask  = std::async(std::launch::async, [&]{ return webp.decodeRGBA(lblV.data,  lblV.size); });
            shNCentroids = centTask.get();
            shNLabels    = lblTask.get();
            if (static_cast<size_t>(shNLabels.width) * static_cast<size_t>(shNLabels.height) <
                static_cast<size_t>(numPoints))
                throw std::runtime_error("SOG shN labels texture too small.");
            shNPaletteCount = shNObj.at("count").get<int>();
            hasShN = true;
        }
    }

    // --- Pre-decode shN centroids into a flat float buffer ---
    const int coeffPerChannel = std::min(shCoeffs, (D+1)*(D+1) - 1);
    const uint8_t* shNLabelsData        = nullptr;
    std::vector<float> shNCentroidDecoded;
    const float*       shNCentroidDecodedData = nullptr;
    size_t shCopyFloatCount = 0, shCopyBytes = 0;

    if (hasShN && shExtraPerPoint > 0 && shCoeffs > 0 && coeffPerChannel > 0) {
        const uint8_t* centData   = shNCentroids.rgba.data();
        shNLabelsData              = shNLabels.rgba.data();
        const int shCW             = shNCentroids.width;
        const int reqH             = (shNPaletteCount + 63) / 64;
        if (shCW < 64 * shCoeffs || shNCentroids.height < reqH)
            throw std::runtime_error("SOG shN centroids texture too small for palette layout.");

        shNCentroidDecoded.assign(
            static_cast<size_t>(shNPaletteCount) * coeffPerChannel * 3, 0.f);
        for (int label = 0; label < shNPaletteCount; ++label) {
            const int cy = label / 64;
            const int cxBase = (label % 64) * shCoeffs;
            for (int coeff = 0; coeff < coeffPerChannel; ++coeff) {
                const size_t centBase = (static_cast<size_t>(cy) * shCW +
                                         static_cast<size_t>(cxBase + coeff)) * 4;
                const size_t decBase  = (static_cast<size_t>(label) * coeffPerChannel +
                                          coeff) * 3;
                shNCentroidDecoded[decBase + 0] = shNCodebook[centData[centBase + 0]];
                shNCentroidDecoded[decBase + 1] = shNCodebook[centData[centBase + 1]];
                shNCentroidDecoded[decBase + 2] = shNCodebook[centData[centBase + 2]];
            }
        }
        shNCentroidDecodedData = shNCentroidDecoded.data();
        shCopyFloatCount = static_cast<size_t>(coeffPerChannel) * 3;
        shCopyBytes      = shCopyFloatCount * sizeof(float);
    }
    tShNPrepDone = Utils::nowUs();

    // --- Build position decode LUT ---
    std::array<std::vector<float>, 3> posLut;
    for (int c = 0; c < 3; ++c) {
        posLut[c].resize(65536);
        const float range = meansMax[c] - meansMin[c];
        if (range <= kQuantEpsilon) {
            std::fill(posLut[c].begin(), posLut[c].end(),
                      invLogTransform(meansMin[c]));
            continue;
        }
        const float step = range / 65535.f;
        for (int q = 0; q <= 65535; ++q)
            posLut[c][q] = invLogTransform(meansMin[c] + step * static_cast<float>(q));
    }

    std::array<float, kSogCodebookSize> decodedScaleCB;
    for (size_t i = 0; i < kSogCodebookSize; ++i)
        decodedScaleCB[i] = std::exp(scaleCodebook[i]);

    std::array<float, 256> opacityLut;
    for (int i = 0; i < 256; ++i)
        opacityLut[i] = static_cast<float>(i) / 255.f;

    if (fileShDegree < D) {
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < numPoints; ++i)
            result.shs[i].fill(0.f);
    }

    const uint8_t* meansLData = meansL.rgba.data();
    const uint8_t* meansUData = meansU.rgba.data();
    const uint8_t* quatsData  = quats.rgba.data();
    const uint8_t* scalesData = scales.rgba.data();
    const uint8_t* sh0Data    = sh0.rgba.data();

    float minX =  FLT_MAX, minY =  FLT_MAX, minZ =  FLT_MAX;
    float maxX = -FLT_MAX, maxY = -FLT_MAX, maxZ = -FLT_MAX;

    #pragma omp parallel for schedule(static) \
        reduction(min:minX,minY,minZ) reduction(max:maxX,maxY,maxZ)
    for (int i = 0; i < numPoints; ++i) {
        const size_t base = static_cast<size_t>(i) * 4;

        const uint16_t qx = static_cast<uint16_t>(meansLData[base+0]) |
                            (static_cast<uint16_t>(meansUData[base+0]) << 8U);
        const uint16_t qy = static_cast<uint16_t>(meansLData[base+1]) |
                            (static_cast<uint16_t>(meansUData[base+1]) << 8U);
        const uint16_t qz = static_cast<uint16_t>(meansLData[base+2]) |
                            (static_cast<uint16_t>(meansUData[base+2]) << 8U);

        const float px = posLut[0][qx];
        const float py = posLut[1][qy];
        const float pz = posLut[2][qz];
        result.points[i][0] = px;
        result.points[i][1] = py;
        result.points[i][2] = pz;
        minX = std::min(minX, px); minY = std::min(minY, py); minZ = std::min(minZ, pz);
        maxX = std::max(maxX, px); maxY = std::max(maxY, py); maxZ = std::max(maxZ, pz);

        result.rot[i] = unpackQuaternion(quatsData + base);

        result.scale[i][0] = decodedScaleCB[scalesData[base+0]];
        result.scale[i][1] = decodedScaleCB[scalesData[base+1]];
        result.scale[i][2] = decodedScaleCB[scalesData[base+2]];

        result.shs[i][0] = sh0Codebook[sh0Data[base+0]];
        result.shs[i][1] = sh0Codebook[sh0Data[base+1]];
        result.shs[i][2] = sh0Codebook[sh0Data[base+2]];
        result.opacity[i] = opacityLut[sh0Data[base+3]];

        if (shNCentroidDecodedData) {
            const int label = static_cast<int>(shNLabelsData[base+0]) |
                              (static_cast<int>(shNLabelsData[base+1]) << 8);
            if (label >= 0 && label < shNPaletteCount) {
                std::memcpy(result.shs[i].data() + 3,
                            shNCentroidDecodedData +
                            static_cast<size_t>(label) * shCopyFloatCount,
                            shCopyBytes);
            } else {
                std::memset(result.shs[i].data() + 3, 0, shCopyBytes);
            }
        }
    }
    tPointLoopDone = Utils::nowUs();

    result.sceneMin = numPoints > 0 ? Eigen::Vector3f(minX, minY, minZ)
                                    : Eigen::Vector3f::Zero();
    result.sceneMax = numPoints > 0 ? Eigen::Vector3f(maxX, maxY, maxZ)
                                    : Eigen::Vector3f::Zero();

    auto tEnd = Utils::nowUs();
    GS_INFO("SOG load breakdown ms: meta=%.2f, texDecode=%.2f, shNPrep=%.2f, "
            "pointLoop=%.2f, tail=%.2f",
            static_cast<float>(tMetaDone      - t0)             / 1000.f,
            static_cast<float>(tDecodeDone    - tMetaDone)      / 1000.f,
            static_cast<float>(tShNPrepDone   - tDecodeDone)    / 1000.f,
            static_cast<float>(tPointLoopDone - tShNPrepDone)   / 1000.f,
            static_cast<float>(tEnd           - tPointLoopDone) / 1000.f);
    GS_INFO("SogLoader: loaded %d gaussians from %s (%.2f ms)",
            numPoints, path.c_str(),
            static_cast<float>(tEnd - t0) / 1000.f);
}


// ======================================================================
// LoaderFactory implementation
// ======================================================================

template <int D>
std::string LoaderFactory<D>::defaultSogCachePath(const std::string& modelPath) {
    const std::filesystem::path mp(modelPath);
    return (mp.parent_path() / ".cache" / (mp.stem().string() + ".sog")).string();
}

template <int D>
bool LoaderFactory<D>::peekInfo(const std::string& modelPath,
                                int*               outShDegree,
                                int*               outNumPoints)
{
    if (hasExtension(modelPath, ".sog"))
        return SogLoader<D>::peekInfo(modelPath, outShDegree, outNumPoints);
    // Default to PLY for everything else
    return PlyLoader<D>::peekInfo(modelPath, outShDegree, outNumPoints);
}

template <int D>
std::unique_ptr<IGaussianLoader<D>> LoaderFactory<D>::create(
    const std::string& modelPath,
    const std::string& sogCachePath,
    bool               rebuildCache)
{
    if (hasExtension(modelPath, ".sog")) {
        // Already an SOG -- load it directly, no caching needed.
        return std::make_unique<SogLoader<D>>();
    }

    // For PLY (and future formats): use SOG cache when available.
    const bool cacheExists = !sogCachePath.empty() &&
                             std::filesystem::exists(sogCachePath);
    if (!rebuildCache && cacheExists) {
        GS_INFO("LoaderFactory: using SOG cache %s", sogCachePath.c_str());
        return std::make_unique<SogLoader<D>>();
    }

    // Cache missing or forced rebuild: load source and write cache.
    const std::string writePath = sogCachePath.empty() ?
        defaultSogCachePath(modelPath) : sogCachePath;
    GS_INFO("LoaderFactory: loading PLY %s, cache -> %s",
            modelPath.c_str(), writePath.c_str());
    return std::make_unique<PlyLoader<D>>(writePath);
}

} // namespace optisplat
