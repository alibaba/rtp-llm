#include "rtp_llm/cpp/media_network/ContentDecoder.h"

#include <brotli/decode.h>
#include <zlib.h>
#include <zstd.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace rtp_llm::media {
namespace {
constexpr size_t kBufferSize = 16384;
constexpr size_t kMaxCodings = 5;  // Existing libcurl chain depth limit.

[[noreturn]] void invalidEncoding() {
    throw std::runtime_error("invalid HTTP content encoding");
}

enum class Coding {
    Unknown,
    Gzip,
    Deflate,
    Brotli,
    Zstd
};

Coding coding(std::string_view token) {
    if (token == "gzip" || token == "x-gzip")
        return Coding::Gzip;
    if (token == "deflate")
        return Coding::Deflate;
    if (token == "br")
        return Coding::Brotli;
    if (token == "zstd")
        return Coding::Zstd;
    return Coding::Unknown;
}

class Output final: public ContentDecoder {
public:
    explicit Output(Sink sink): sink_(std::move(sink)) {}
    void write(std::string_view bytes) override {
        sink_(bytes);
    }
    void finish() override {}

private:
    Sink sink_;
};

class Layer: public ContentDecoder {
public:
    Layer(std::unique_ptr<ContentDecoder> next, Checkpoint check): next_(std::move(next)), check_(std::move(check)) {}
    void finish() override {
        next_->finish();
    }

protected:
    void emit(size_t count) {
        check_();
        if (count)
            next_->write({buffer_.data(), count});
    }
    std::unique_ptr<ContentDecoder> next_;
    Checkpoint                      check_;
    std::array<char, kBufferSize>   buffer_;
};

class ZlibLayer final: public Layer {
public:
    ZlibLayer(std::unique_ptr<ContentDecoder> next, Checkpoint check, bool gzip, size_t max_probe):
        Layer(std::move(next), std::move(check)), max_probe_(max_probe), gzip_(gzip), probe_(!gzip) {
        if (inflateInit2(&z_, gzip ? MAX_WBITS + 16 : MAX_WBITS) != Z_OK)
            invalidEncoding();
    }
    ~ZlibLayer() override {
        inflateEnd(&z_);
    }

    void write(std::string_view bytes) override {
        if (done_ || bytes.empty())
            return;
        // urllib3 retries raw DEFLATE from the beginning until wrapped
        // DEFLATE produces its first output, even across input chunks.
        std::string replay;
        if (probe_ && bytes.size() > max_probe_ - prefix_.size()) {
            // An ambiguous no-output prefix must not retain unbounded input.
            // Continue with the wrapped stream once its replay budget is spent.
            probe_ = false;
            std::string{}.swap(prefix_);
        }
        if (probe_)
            prefix_.append(bytes);
        z_.next_in  = reinterpret_cast<Bytef*>(const_cast<char*>(bytes.data()));
        z_.avail_in = static_cast<uInt>(bytes.size());
        do {
            z_.next_out  = reinterpret_cast<Bytef*>(buffer_.data());
            z_.avail_out = buffer_.size();
            int rc       = inflate(&z_, Z_NO_FLUSH);
            if (rc == Z_DATA_ERROR && probe_) {
                probe_ = false;
                replay.swap(prefix_);
                if (inflateReset2(&z_, -MAX_WBITS) != Z_OK)
                    invalidEncoding();
                z_.next_in  = reinterpret_cast<Bytef*>(replay.data());
                z_.avail_in = static_cast<uInt>(replay.size());
                continue;
            }
            if (rc != Z_OK && rc != Z_BUF_ERROR && rc != Z_STREAM_END) {
                if (rc == Z_DATA_ERROR && another_member_) {
                    done_ = true;  // Ignore trailing non-gzip data after a complete member.
                    return;
                }
                invalidEncoding();
            }
            size_t count = buffer_.size() - z_.avail_out;
            if (count && probe_) {
                probe_ = false;
                std::string{}.swap(prefix_);
            }
            emit(count);
            if (rc == Z_STREAM_END) {
                if (!gzip_) {
                    done_ = true;
                    return;
                }
                another_member_ = true;
                if (inflateReset(&z_) != Z_OK)
                    invalidEncoding();
            } else if (rc == Z_BUF_ERROR) {
                return;
            }
        } while (z_.avail_in || !z_.avail_out);
    }

private:
    z_stream     z_{};
    const size_t max_probe_;
    const bool   gzip_;
    bool         probe_, another_member_ = false, done_ = false;
    std::string  prefix_;
};

class BrotliLayer final: public Layer {
public:
    BrotliLayer(std::unique_ptr<ContentDecoder> next, Checkpoint check): Layer(std::move(next), std::move(check)) {
        if (!state_)
            invalidEncoding();
    }
    void write(std::string_view bytes) override {
        auto*  input     = reinterpret_cast<const uint8_t*>(bytes.data());
        size_t remaining = bytes.size();
        for (;;) {
            size_t space = buffer_.size();
            auto*  out   = reinterpret_cast<uint8_t*>(buffer_.data());
            auto   rc    = BrotliDecoderDecompressStream(state_.get(), &remaining, &input, &space, &out, nullptr);
            if (rc == BROTLI_DECODER_RESULT_ERROR)
                invalidEncoding();
            emit(buffer_.size() - space);
            if (rc != BROTLI_DECODER_RESULT_NEEDS_MORE_OUTPUT)
                return;
        }
    }

private:
    std::unique_ptr<BrotliDecoderState, decltype(&BrotliDecoderDestroyInstance)> state_{
        BrotliDecoderCreateInstance(nullptr, nullptr, nullptr), BrotliDecoderDestroyInstance};
};

class ZstdLayer final: public Layer {
public:
    ZstdLayer(std::unique_ptr<ContentDecoder> next, Checkpoint check, bool check_eof):
        Layer(std::move(next), std::move(check)), check_eof_(check_eof) {
        if (!state_)
            invalidEncoding();
    }
    void write(std::string_view bytes) override {
        ZSTD_inBuffer input{bytes.data(), bytes.size(), 0};
        for (;;) {
            ZSTD_outBuffer out{buffer_.data(), buffer_.size(), 0};
            auto           remaining = ZSTD_decompressStream(state_.get(), &out, &input);
            if (ZSTD_isError(remaining))
                invalidEncoding();
            complete_ = remaining == 0;
            emit(out.pos);
            if (input.pos == input.size && (complete_ || out.pos < out.size))
                return;
        }
    }
    void finish() override {
        // urllib3 MultiDecoder.flush finalizes the innermost decoder.
        if (check_eof_ && !complete_)
            invalidEncoding();
        Layer::finish();
    }

private:
    std::unique_ptr<ZSTD_DCtx, decltype(&ZSTD_freeDCtx)> state_{ZSTD_createDCtx(), ZSTD_freeDCtx};
    bool                                                 check_eof_, complete_ = false;
};
}  // namespace

std::unique_ptr<ContentDecoder>
ContentDecoder::create(std::string_view header, Sink output, Checkpoint checkpoint, size_t byte_limit) {
    // HTTPResponse._init_decoder leaves wholly unknown codings untouched.
    // MultiDecoder trims list items and uses DEFLATE for any unknown item.
    bool                            list = header.find(',') != header.npos;
    std::array<Coding, kMaxCodings> codings;
    size_t                          count      = 0;
    bool                            recognized = false;
    for (;;) {
        auto comma = header.find(',');
        auto token = header.substr(0, comma);
        if (list) {
            auto first = token.find_first_not_of(" \t");
            token      = first == token.npos ? std::string_view{} :
                                               token.substr(first, token.find_last_not_of(" \t") - first + 1);
        }
        auto kind = coding(token);
        recognized |= kind != Coding::Unknown;
        if (count < codings.size())
            codings[count] = kind;
        ++count;
        if (comma == header.npos)
            break;
        header.remove_prefix(comma + 1);
    }
    if (!recognized)
        return {};
    if (count > codings.size())
        invalidEncoding();
    // At least one full network read, and small enough for zlib's avail_in.
    const size_t max_probe = std::min<size_t>(std::numeric_limits<uInt>::max(), std::max<size_t>(65536, byte_limit));
    std::unique_ptr<ContentDecoder> decoder = std::make_unique<Output>(std::move(output));
    for (size_t i = 0; i < count; ++i) {
        switch (codings[i]) {
            case Coding::Gzip:
            case Coding::Deflate:
            case Coding::Unknown:
                decoder =
                    std::make_unique<ZlibLayer>(std::move(decoder), checkpoint, codings[i] == Coding::Gzip, max_probe);
                break;
            case Coding::Brotli:
                decoder = std::make_unique<BrotliLayer>(std::move(decoder), checkpoint);
                break;
            case Coding::Zstd:
                decoder = std::make_unique<ZstdLayer>(std::move(decoder), checkpoint, i == 0);
                break;
        }
    }
    return decoder;
}
}  // namespace rtp_llm::media
