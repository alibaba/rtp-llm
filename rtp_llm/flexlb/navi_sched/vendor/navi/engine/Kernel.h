#pragma once

#include "navi/builder/KernelDefBuilder.h"
#include "navi/engine/KernelComputeContext.h"
#include "navi/engine/KernelConfigContext.h"
#include "navi/engine/KernelCreator.h"
#include "navi/engine/KernelInitContext.h"
#include "navi/log/NaviLogger.h"

namespace navi {

class NodeDef;
class Biz;

class Kernel
{
public:
    Kernel();
    virtual ~Kernel();
private:
    Kernel(const Kernel &);
    Kernel& operator=(const Kernel &);
public:
    virtual void def(KernelDefBuilder &builder) const = 0;
    virtual bool config(KernelConfigContext &ctx) {
        return true;
    }
    virtual ErrorCode init(KernelInitContext &ctx) {
        return EC_NONE;
    }
    virtual ErrorCode compute(KernelComputeContext &ctx) = 0;
    virtual bool isExpensive() const {
        return true;
    }
    virtual void __navi__declare_depend(navi::KernelDefBuilder &builder) {
    }
    inline void __navi__declare_depend_t(navi::DependMatcher<-2> *matcher, navi::KernelDefBuilder &builder) {}
public:
    const std::string &getNodeName() const;
    const std::string &getKernelName() const;
    const std::string &getDeviceName() const;
public:
    template <typename T, std::enable_if_t<std::is_base_of_v<navi::Kernel, T>, bool> = true>
    static T *navi_bind_cast(void *obj) {
        return dynamic_cast<T *>(static_cast<navi::Kernel *>(obj));
    }
private:
    void setNodeDef(const NodeDef *def);
    friend class Node;
private:
    const NodeDef *_def;
};

NAVI_TYPEDEF_PTR(Kernel);

#define REGISTER_KERNEL(NAME) REGISTER_KERNEL_HELPER(__COUNTER__, NAME)
#define REGISTER_KERNEL_HELPER(ctr, NAME) REGISTER_KERNEL_UNIQ(ctr, NAME, false)
#define REGISTER_KERNEL_UNIQ(ctr, NAME, naviRegistry)                          \
    __attribute__((constructor)) static void Register##NAME##Kernel##ctr() {   \
        auto ret = ::navi::CreatorRegistry::current(                           \
            navi::RT_KERNEL) -> addCreatorFunc([]() -> ::navi::Creator * {     \
            static_assert(std::is_base_of<navi::Kernel, NAME>::value,          \
                          "illegal kernel register type, must inherit "        \
                          "navi::Kernel, class: " #NAME);                      \
            static_assert(!std::is_base_of<navi::Resource, NAME>::value,       \
                          "illegal kernel register type, kernel can't "        \
                          "inherit navi::Resource, class: " #NAME);            \
            auto def = new navi::KernelDef();                                  \
            navi::KernelDefBuilder builder(def);                               \
            NAME thisKernel;                                                   \
            thisKernel.def(builder);                                           \
            NAVI_KERNEL_LOG(DEBUG,                                             \
                            "begin create depend, file: %s, NAME: %s",         \
                            __FILE__,                                          \
                            #NAME)                                             \
            thisKernel.__navi__declare_depend(builder);                        \
            auto newInfo = def->mutable_new_info();                            \
            newInfo->set_new_size(sizeof(NAME));                               \
            newInfo->set_align_size(alignof(NAME));                            \
            return new ::navi::KernelCreator(                                  \
                __FILE__,                                                      \
                __LINE__,                                                      \
                naviRegistry,                                                  \
                builder,                                                       \
                [](void *addr) {                                               \
                    auto ptr = new (addr) NAME();                              \
                    return ptr;                                                \
                },                                                             \
                [](navi::Kernel *other, int64_t &castOffset) {                 \
                    NAME *castValue = dynamic_cast<NAME *>(other);             \
                    if (castValue) {                                           \
                        castOffset = (((char *)castValue) - ((char *)other));  \
                        return true;                                           \
                    } else {                                                   \
                        return false;                                          \
                    }                                                          \
                });                                                            \
        });                                                                    \
        if (!ret) {                                                            \
            NAVI_KERNEL_LOG(ERROR, "register kernel [%s] failed", #NAME);      \
        }                                                                      \
    }

#define REGISTER_KERNEL_INIT(FUNC)                                             \
    REGISTER_KERNEL_INIT_HELPER(__COUNTER__, FUNC)
#define REGISTER_KERNEL_INIT_HELPER(ctr, FUNC)                                 \
    REGISTER_KERNEL_INIT_UNIQ(ctr, FUNC)
#define REGISTER_KERNEL_INIT_UNIQ(ctr, FUNC)                                                                           \
    __attribute__((constructor)) static void Register##KernelInit##ctr() {                                             \
        auto ret = ::navi::CreatorRegistry::current(navi::RT_KERNEL)->addModuleInitFunc([]() {                         \
            if (!FUNC()) {                                                                                             \
                NAVI_KERNEL_LOG(ERROR, "module init failed");                                                          \
                return false;                                                                                          \
            }                                                                                                          \
            return true;                                                                                               \
        });                                                                                                            \
        if (!ret) {                                                                                                    \
            NAVI_KERNEL_LOG(ERROR, "register ModuleInitFunc [%s] failed", #FUNC);                                      \
        }                                                                                                              \
    }
}

#define KERNEL_DEPEND_DECLARE()                                                                                        \
public:                                                                                                                \
    __attribute__((always_inline)) void __navi__declare_depend(navi::KernelDefBuilder &builder) override {             \
        navi::DependMatcher<512> *matcherEnd = nullptr;                                                                \
        __navi__declare_depend_t(matcherEnd, builder);                                                                 \
    }                                                                                                                  \
                                                                                                                       \
protected:                                                                                                             \
    template <int N>                                                                                                   \
    __attribute__((always_inline)) void __navi__declare_depend_t(navi::DependMatcher<N> *matcher,                      \
                                                                 navi::KernelDefBuilder &builder) {                    \
        __navi__declare_depend_t((navi::DependMatcher<N - 1> *)0, builder);                                            \
    }                                                                                                                  \
    using navi::Kernel::__navi__declare_depend_t;

#define KERNEL_DEPEND_DECLARE_BASE(BASE)                                                                               \
public:                                                                                                                \
    __attribute__((always_inline)) void __navi__declare_depend(navi::KernelDefBuilder &builder) override {             \
        BASE::__navi__declare_depend(builder);                                                                         \
        navi::DependMatcher<512> *matcherEnd = nullptr;                                                                \
        __navi__declare_depend_t(matcherEnd, builder);                                                                 \
    }                                                                                                                  \
                                                                                                                       \
protected:                                                                                                             \
    template <int N>                                                                                                   \
    __attribute__((always_inline)) void __navi__declare_depend_t(navi::DependMatcher<N> *matcher,                      \
                                                                 navi::KernelDefBuilder &builder) {                    \
        __navi__declare_depend_t((navi::DependMatcher<N - 1> *)0, builder);                                            \
    }                                                                                                                  \
                                                                                                                       \
protected:                                                                                                             \
    using navi::Kernel::__navi__declare_depend_t;

#define KERNEL_DEPEND_ON(CLASS, NAME) KERNEL_DEPEND_ON_ID(CLASS, NAME, CLASS::RESOURCE_ID, true)
#define KERNEL_DEPEND_ON_FALSE(CLASS, NAME) KERNEL_DEPEND_ON_ID(CLASS, NAME, CLASS::RESOURCE_ID, false)

#define KERNEL_DEPEND_ON_ID(CLASS, NAME, ID, REQUIRE)                                                                  \
    KERNEL_DECLARE_DEPEND_ON_HELPER(__COUNTER__, CLASS, NAME, ID, REQUIRE)
#define KERNEL_DECLARE_DEPEND_ON_HELPER(ctr, CLASS, NAME, ID, REQUIRE)                                                 \
    KERNEL_DECLARE_DEPEND_ON_UNIQ(ctr, CLASS, NAME, ID, REQUIRE)
#define KERNEL_DECLARE_DEPEND_ON_UNIQ(ctr, CLASS, NAME, ID, REQUIRE)                                                   \
    union {                                                                                                            \
        CLASS *const NAME = nullptr;                                                                                   \
        CLASS *NAME##__;                                                                                         \
    };                                                                                                                 \
    navi::StaticResourceBindFunc __navi_kernel_resource_binder_func_##NAME() {                                         \
        return BIND_RESOURCE_TO(NAME##__);                                                                       \
    }                                                                                                                  \
    __attribute__((always_inline)) void __navi__declare_depend_t(navi::DependMatcher<ctr> *matcher,                    \
                                                                 navi::KernelDefBuilder &builder) {                    \
        NAVI_KERNEL_LOG(DEBUG, "kernel depend, file: %s, ID: %s, NAME: %s, ctr: %d", __FILE__, #ID, #NAME, ctr);       \
        builder.dependOn(ID, REQUIRE, __navi_kernel_resource_binder_func_##NAME());                                    \
        navi::DependMatcher<ctr - 1> *nextMatcher = nullptr;                                                           \
        __navi__declare_depend_t(nextMatcher, builder);                                                                \
    }

#define KERNEL_NAMED_DATA(CLASS, NAME, ID) KERNEL_DECLARE_NAMED_DATA_HELPER(__COUNTER__, CLASS, NAME, ID, true)
#define KERNEL_NAMED_DATA_FALSE(CLASS, NAME, ID) KERNEL_DECLARE_NAMED_DATA_HELPER(__COUNTER__, CLASS, NAME, ID, false)

#define KERNEL_DECLARE_NAMED_DATA_HELPER(ctr, CLASS, NAME, ID, REQUIRE)                                                \
    KERNEL_DECLARE_NAMED_DATA_UNIQ(ctr, CLASS, NAME, ID, REQUIRE)
#define KERNEL_DECLARE_NAMED_DATA_UNIQ(ctr, CLASS, NAME, ID, REQUIRE)                                                  \
    union {                                                                                                            \
        CLASS *const NAME = nullptr;                                                                                   \
        CLASS *NAME##__;                                                                                               \
    };                                                                                                                 \
    navi::NamedDataBindFunc __navi_kernel_named_data_binder_func_##NAME() { return BIND_DATA_TO(NAME##__); }           \
    __attribute__((always_inline)) void __navi__declare_depend_t(navi::DependMatcher<ctr> *matcher,                    \
                                                                 navi::KernelDefBuilder &builder) {                    \
        builder.namedData(ID, REQUIRE, __navi_kernel_named_data_binder_func_##NAME());                                 \
        navi::DependMatcher<ctr - 1> *nextMatcher = nullptr;                                                           \
        __navi__declare_depend_t(nextMatcher, builder);                                                                \
    }
