#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "rtp_llm/cpp/multimodal_processor/MultimodalProcessor.h"

namespace rtp_llm {
namespace {

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

torch::Tensor readFeature(const char* path, int64_t rows) {
    auto feature = torch::empty({rows, 7168}, torch::kBFloat16);
    std::ifstream file(path, std::ios::binary);
    require(file.is_open(), std::string("cannot open feature: ") + path);
    const auto bytes = static_cast<std::streamsize>(feature.nbytes());
    file.read(static_cast<char*>(feature.data_ptr()), bytes);
    require(file.gcount() == bytes && file.peek() == std::char_traits<char>::eof(),
            std::string("feature byte count differs from shape: ") + path);
    return feature;
}

torch::Tensor readTokenIds(const char* path, int64_t count) {
    require(count > 0, "input token count must be positive");
    auto          ids = torch::empty({count}, torch::kInt32);
    std::ifstream file(path, std::ios::binary);
    require(file.is_open(), std::string("cannot open token ids: ") + path);
    const auto bytes = static_cast<std::streamsize>(ids.nbytes());
    file.read(static_cast<char*>(ids.data_ptr()), bytes);
    require(file.gcount() == bytes && file.peek() == std::char_traits<char>::eof(),
            std::string("token id byte count differs from shape: ") + path);
    return ids;
}

class CheckProcessor final: public MultimodalProcessor {
public:
    CheckProcessor(int32_t media_pad_id, std::vector<torch::Tensor> features):
        MultimodalProcessor(py::object(), MMModelConfig{true, {{media_pad_id}}, false}, 131072),
        features_(std::move(features)) {}

private:
    ErrorResult<MultimodalOutput> MultimodalEmbedding(const std::vector<MultimodalInput> inputs,
                                                      std::string /* ip_port */ = "") override {
        require(inputs.size() == features_.size(), "media count differs from feature count");
        for (const auto& input : inputs) {
            require(!input.url.empty() && input.mm_type == 1, "expected ordered image media inputs");
        }
        MultimodalOutput output;
        output.mm_features = features_;
        return output;
    }

    std::vector<torch::Tensor> features_;
};

std::shared_ptr<GenerateInput>
expand(CheckProcessor& processor, const torch::Tensor& token_ids, const std::vector<MultimodalInput>& media) {
    auto input       = std::make_shared<GenerateInput>();
    input->input_ids = token_ids.clone();
    input->multimodal_inputs = media;
    const auto status        = processor.updateMultimodalFeatures(input);
    require(status.ok(), std::string("feature expansion failed: ") + status.ToString());
    return input;
}

void checkExpansion(const std::shared_ptr<GenerateInput>& input,
                    const torch::Tensor&                   original_ids,
                    int32_t                                media_pad_id,
                    const std::vector<torch::Tensor>&       features) {
    int64_t total = original_ids.numel() - static_cast<int64_t>(features.size());
    for (const auto& feature : features) {
        total += feature.size(0);
    }
    require(input->input_ids.numel() == total, "expanded token length is wrong");
    require(input->text_tokens_mask && input->mm_locs && input->multimodal_features, "feature metadata is missing");
    require(input->mm_locs.value().numel() == static_cast<int64_t>(features.size()),
            "media feature location count is wrong");
    const auto* old_ids = original_ids.data_ptr<int32_t>();
    const auto* ids     = input->input_ids.data_ptr<int32_t>();
    const auto* mask    = input->text_tokens_mask.value().data_ptr<int32_t>();
    const auto* locs    = input->mm_locs.value().data_ptr<int32_t>();
    int64_t     cursor  = 0;
    size_t      image   = 0;
    for (int64_t i = 0; i < original_ids.numel(); ++i) {
        if (old_ids[i] == media_pad_id) {
            require(image < features.size(), "more image placeholders than features");
            require(locs[image] == cursor, "media feature location is wrong");
            for (int64_t row = 0; row < features[image].size(0); ++row) {
                require(mask[cursor++] == 0, "media row is marked as text");
            }
            ++image;
        } else {
            require(ids[cursor] == old_ids[i], "surrounding text token changed during expansion");
            require(mask[cursor++] == 1, "text token is marked as media");
        }
    }
    require(image == features.size() && cursor == total, "expanded feature order or length is wrong");
    const auto& returned = input->multimodal_features.value();
    require(returned.size() == features.size(), "expanded feature count is wrong");
    for (size_t i = 0; i < features.size(); ++i) {
        require(torch::equal(returned[i], features[i]), "processor altered the real feature tensor");
    }
}

}  // namespace
}  // namespace rtp_llm

int main(int argc, char** argv) {
    using namespace rtp_llm;
    try {
        require(argc >= 12 && (argc - 4) % 4 == 0,
                "usage: check <media_pad_id> <token_ids.raw> <token_count> <url0> <type0> <image0.raw> <rows0> ...");
        const auto pad_id = static_cast<int32_t>(std::stoi(argv[1]));
        auto       token_ids = readTokenIds(argv[2], std::stoll(argv[3]));
        auto       features  = std::vector<torch::Tensor>{};
        auto       media     = std::vector<MultimodalInput>{};
        for (int i = 4; i < argc; i += 4) {
            media.emplace_back(argv[i], std::stoi(argv[i + 1]));
            features.push_back(readFeature(argv[i + 2], std::stoll(argv[i + 3])));
        }
        require(features[0].size(0) != features[1].size(0), "use different image spans to detect reordering");

        CheckProcessor processor(pad_id, features);
        auto           first  = expand(processor, token_ids, media);
        auto           second = expand(processor, token_ids, media);
        checkExpansion(first, token_ids, pad_id, features);
        checkExpansion(second, token_ids, pad_id, features);
        require(torch::equal(first->input_ids, second->input_ids), "equal features produced different hash tokens");

        auto changed_features = features;
        changed_features[0]   = features[0].clone();
        static_cast<uint8_t*>(changed_features[0].data_ptr())[0] ^= 1;
        CheckProcessor changed_processor(pad_id, changed_features);
        auto           changed_input = expand(changed_processor, token_ids, media);
        const auto     first_loc = first->mm_locs.value().data_ptr<int32_t>()[0];
        require(first->input_ids.data_ptr<int32_t>()[first_loc]
                    != changed_input->input_ids.data_ptr<int32_t>()[first_loc],
                "changing a feature row did not change its hash token");
        for (int64_t i = 0; i < first->input_ids.numel(); ++i) {
            if (i != first_loc) {
                require(first->input_ids.data_ptr<int32_t>()[i] == changed_input->input_ids.data_ptr<int32_t>()[i],
                        "changing one feature row changed another token");
            }
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << std::endl;
        return 1;
    }
}
