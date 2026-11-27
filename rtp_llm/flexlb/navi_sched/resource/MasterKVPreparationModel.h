#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>

namespace navi_sched {

// Four fixed preparation features: bias, read tokens/4096, earlier preparing
// read tokens/4096, initial predicted compute queue/1000ms. Existing node runtime
// locking owns this model. No fitting, allocation or lock occurs in predict().
class MasterKVPreparationModel {
public:
    static constexpr size_t FEATURE_COUNT = 4;
    using Features = std::array<double, FEATURE_COUNT>;

    bool record(const Features &features, double preparationMs) {
        if (!validFeatures(features) || !std::isfinite(preparationMs) || preparationMs < 0.0)
            return false;
        for (size_t i = 0; i < FEATURE_COUNT; ++i) {
            _xty[i] += static_cast<long double>(features[i]) * preparationMs;
            for (size_t j = 0; j < FEATURE_COUNT; ++j)
                _xtx[i][j] += static_cast<long double>(features[i]) * features[j];
        }
        ++_samples;
        solve();
        return true;
    }

    std::optional<double> predict(const Features &features) const {
        if (!_samples || !validFeatures(features)) return std::nullopt;
        long double result = 0.0;
        for (size_t i = 0; i < FEATURE_COUNT; ++i) result += _coefficients[i] * features[i];
        const auto totalMs = static_cast<double>(result);
        if (!std::isfinite(totalMs) || totalMs <= 0.0) return std::nullopt;
        return totalMs;
    }

    uint64_t samples() const { return _samples; }
    Features coefficients() const {
        Features result{};
        for (size_t i=0;i<FEATURE_COUNT;++i) result[i]=static_cast<double>(_coefficients[i]);
        return result;
    }

private:
    static bool validFeatures(const Features &features) {
        if (features[0] != 1.0) return false;
        for (auto value : features)
            if (!std::isfinite(value) || value < 0.0) return false;
        return true;
    }

    // Enumerate the 16 active sets of this fixed four-coordinate NNLS problem.
    // Column normalization makes rank detection independent of token units.
    // This is an exact small least-squares solve, not an optimizer update policy.
    void solve() {
        std::array<long double, FEATURE_COUNT> best{};
        long double bestObjective = 0.0; // ||Xb||^2 - 2 y'Xb; y'y is constant.
        for (unsigned mask = 1; mask < (1u << FEATURE_COUNT); ++mask) {
            std::array<size_t, FEATURE_COUNT> indexes{};
            size_t n=0;
            for(size_t i=0;i<FEATURE_COUNT;++i) if(mask&(1u<<i)) indexes[n++]=i;
            long double a[FEATURE_COUNT][FEATURE_COUNT+1]{};
            std::array<long double, FEATURE_COUNT> scale{};
            bool usable=true;
            for(size_t i=0;i<n;++i) {
                scale[i]=std::sqrt(_xtx[indexes[i]][indexes[i]]);
                if (!(scale[i]>0.0L)) {usable=false;break;}
            }
            if(!usable) continue;
            for(size_t i=0;i<n;++i) {
                for(size_t j=0;j<n;++j) a[i][j]=_xtx[indexes[i]][indexes[j]]/(scale[i]*scale[j]);
                a[i][n]=_xty[indexes[i]]/scale[i];
            }
            for(size_t col=0;col<n;++col) {
                size_t pivot=col;
                for(size_t row=col+1;row<n;++row)
                    if(std::abs(a[row][col])>std::abs(a[pivot][col])) pivot=row;
                if(std::abs(a[pivot][col])<=1e-12L) {usable=false;break;}
                if(pivot!=col) for(size_t j=col;j<=n;++j) std::swap(a[pivot][j],a[col][j]);
                const auto diagonal=a[col][col];
                for(size_t j=col;j<=n;++j) a[col][j]/=diagonal;
                for(size_t row=0;row<n;++row) if(row!=col) {
                    const auto factor=a[row][col];
                    for(size_t j=col;j<=n;++j) a[row][j]-=factor*a[col][j];
                }
            }
            if(!usable) continue;
            std::array<long double, FEATURE_COUNT> candidate{};
            for(size_t i=0;i<n;++i) {
                candidate[indexes[i]]=a[i][n]/scale[i];
                if(!std::isfinite(candidate[indexes[i]]) || candidate[indexes[i]]<0.0L) {usable=false;break;}
            }
            if(!usable) continue;
            long double objective=0.0L;
            for(size_t i=0;i<FEATURE_COUNT;++i) {
                objective-=2.0L*candidate[i]*_xty[i];
                for(size_t j=0;j<FEATURE_COUNT;++j) objective+=candidate[i]*_xtx[i][j]*candidate[j];
            }
            if(std::isfinite(objective) && objective<bestObjective) {bestObjective=objective;best=candidate;}
        }
        _coefficients=best;
    }

    uint64_t _samples=0;
    std::array<std::array<long double, FEATURE_COUNT>, FEATURE_COUNT> _xtx{};
    std::array<long double, FEATURE_COUNT> _xty{};
    std::array<long double, FEATURE_COUNT> _coefficients{};
};
} // namespace navi_sched
