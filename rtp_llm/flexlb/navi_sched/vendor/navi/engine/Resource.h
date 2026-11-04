#pragma once

#include "navi/builder/ResourceDefBuilder.h"
#include "navi/engine/ResourceConfigContext.h"
#include "navi/engine/ResourceCreator.h"
#include "navi/engine/ResourceInitContext.h"
#include "navi/proto/GraphDef.pb.h"
#include "navi/log/NaviLogger.h"

namespace navi {

class ResourceMap;

struct StaticGraphInfo {
    std::string name;
    std::string meta;
    GraphDef def;
};

NAVI_TYPEDEF_PTR(StaticGraphInfo);

class Resource : public std::enable_shared_from_this<Resource> {
public:
    Resource() {}
    virtual ~Resource() {}
public:
    virtual void def(ResourceDefBuilder &builder) const = 0;
    virtual bool config(ResourceConfigContext &ctx) {
        return true;
    }
    virtual ErrorCode init(ResourceInitContext &ctx) = 0;
    virtual bool getMeta(std::string &meta) const {
        return false;
    }
    virtual bool getStaticGraphInfo(std::vector<StaticGraphInfoPtr> &graphInfos) const {
        return false;
    }
    virtual SummaryErrorCode getSummary(const std::string &query, SummaryBuilder &builder) const {
        return SEC_NOT_SUPPORT;
    }
    virtual bool isReady() const {
        return true;
    }
    const std::string &getName() {
        if (_name.empty()) {
            initName();
        }
        return _name;
    }
    void initName();
    void setCastPtr(void *ptr) {
        _castPtr = ptr;
    }
    void *getCastPtr() const {
        return _castPtr;
    }
public:
    virtual void __navi__declare_depend(navi::ResourceDefBuilder &builder) {
    }
    inline void __navi__declare_depend_t(navi::DependMatcher<-2> *matcher, navi::ResourceDefBuilder &builder) {}
public:
    template<typename T, std::enable_if_t<std::is_base_of_v<navi::Resource, T>, bool> = true>
    static T *navi_bind_cast(void *obj) {
        return dynamic_cast<T *>(static_cast<navi::Resource *>(obj));
    }
private:
    void setName(const std::string &name) {
        _name = name;
    }
private:
    virtual bool setAsyncPipe(const AsyncPipePtr &pipe) {
        return true;
    }
protected:
    void initLogger(NaviObjectLogger &logger, const std::string &prefix);
    const ResourceMap &getDepends() const {
        return _dependMap;
    }
private:
    friend class ResourceCreator;
    friend class ResourceManager;
    friend class ResourceData;
    friend class ResourceCreateKernel;
    friend class NaviResourceHelper;
    friend class ExternalResource;
private:
    void *_castPtr = nullptr;
    std::string _name;
    ResourceMap _dependMap;
};

NAVI_TYPEDEF_PTR(Resource);

class ExternalResource : public Resource {
public:
    ExternalResource(const std::string &name, ResourceStage stage);
    void def(ResourceDefBuilder &builder) const override final;
    bool config(ResourceConfigContext &ctx) override final {
        return true;
    }
    ErrorCode init(ResourceInitContext &ctx) override final;
private:
    ResourceStage _stage;
};

template <typename T>
class AsyncResourceT : public T {
public:
    AsyncResourceT() {
        static_assert(std::is_base_of<Resource, T>::value, "async resource template is not derived from Resource");
    }
public:
    void def(ResourceDefBuilder &builder) const override final {
        builder.setAsync(true);
        asyncDef(builder);
    }
    ErrorCode init(ResourceInitContext &ctx) override final {
        return asyncInit(ctx);
    }
    virtual void asyncDef(ResourceDefBuilder &builder) const = 0;
    virtual ErrorCode asyncInit(ResourceInitContext &ctx) = 0;
    void notifyReady() {
        autil::ScopedLock lock(_lock);
        if (_ready) {
            return;
        }
        _ready = true;
        if (_pipe) {
            _pipe->setData(nullptr);
        }
    }
    bool isReady() const override final {
        autil::ScopedLock lock(_lock);
        return _ready;
    }
private:
    bool setAsyncPipe(const AsyncPipePtr &pipe) override {
        autil::ScopedLock lock(_lock);
        if (_pipe) {
            return false;
        }
        if (!pipe) {
            return false;
        }
        _pipe = pipe;
        if (_ready) {
            _pipe->setData(nullptr);
        }
        return true;
    };
private:
    friend class ResourceCreateKernel;
private:
    mutable autil::ThreadMutex _lock;
    bool _ready = false;
    AsyncPipePtr _pipe;
};

typedef AsyncResourceT<navi::Resource> AsyncResource;

#define REGISTER_RESOURCE(NAME) REGISTER_RESOURCE_HELPER(__COUNTER__, NAME)
#define REGISTER_RESOURCE_HELPER(ctr, NAME) REGISTER_RESOURCE_UNIQ(ctr, NAME, false)
#define REGISTER_RESOURCE_UNIQ(ctr, NAME, naviRegistry)                        \
    __attribute__((constructor)) static void Register##NAME##Resource() {      \
        auto ret = ::navi::CreatorRegistry::current(                           \
            navi::RT_RESOURCE) -> addCreatorFunc([]() -> ::navi::Creator * {   \
            static_assert(std::is_base_of<navi::Resource, NAME>::value,        \
                          "illegal resource register type, must inherit "      \
                          "navi::Resource, class: " #NAME);                    \
            static_assert(!std::is_base_of<navi::Kernel, NAME>::value,         \
                          "illegal resource register type, resource "          \
                          "can't inherit navi::Kernel, class: " #NAME);        \
            auto def = new navi::ResourceDef();                                \
            navi::ResourceDefBuilder builder(def);                             \
            NAME thisResource;                                                 \
            thisResource.def(builder);                                         \
            thisResource.__navi__declare_depend(builder);                      \
            return new ::navi::ResourceCreator(                                \
                __FILE__,                                                      \
                __LINE__,                                                      \
                naviRegistry,                                                  \
                builder,                                                       \
                []() {                                                         \
                    auto ptr = new NAME();                                     \
                    navi::ResourcePtr resource(ptr);                           \
                    return resource;                                           \
                },                                                             \
                [](navi::Resource *other, int64_t &castOffset) {               \
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
            NAVI_KERNEL_LOG(ERROR, "register resource [%s] failed", #NAME);    \
        }                                                                      \
    }

#define RESOURCE_ASSERT(resource)                                              \
    if (!resource) {                                                           \
        return navi::EC_LACK_RESOURCE;                                         \
    }

template <typename T>
bool bindDynamicResourceTo(navi::Resource *depend, std::set<T *> &member, const char *&typeName) {
    static_assert(std::is_base_of<navi::Resource, T>::value,
                  "illegal member type, use sub class of navi::Resource");
    static_assert(!std::is_same<navi::Resource, T>::value,
                  "illegal member type, can't use navi::Resource for match");
    T *typedDepend;
    auto castDepend = depend->getCastPtr();
    if (castDepend) {
        typedDepend = (T *)castDepend;
        assert(typedDepend == dynamic_cast<T *>(depend));
    } else {
        typedDepend = dynamic_cast<T *>(depend);
    }
    if (!typedDepend) {
        typeName = typeid(T).name();
        return false;
    }
    member.insert(typedDepend);
    return true;
}

template <typename T>
void dynamicResourceCastOffset(navi::Resource *depend,
                               int64_t *dependOffset) {
    static_assert(std::is_base_of<navi::Resource, T>::value,
                  "illegal member type, use sub class of navi::Resource");
    static_assert(!std::is_same<navi::Resource, T>::value,
                  "illegal member type, can't use navi::Resource for match");
    auto *typedDepend = dynamic_cast<T *>(depend);
    if (!typedDepend) {
        *dependOffset = navi::NAVI_INVALID_CAST_OFFSET;
    } else {
        auto offsetDepend = (((char *)typedDepend) - ((char *)depend));
        navi::updateCastOffset(offsetDepend, *dependOffset);
    }
}

#define CAST_OFFSET_OBJ()                                                      \
    T *typedObj = nullptr;                                                     \
    if (navi::NAVI_INVALID_CAST_OFFSET != *objOffset) {                        \
        typedObj = navi_bind_cast<T>(obj);                                     \
        if (!typedObj) {                                                       \
            *objOffset = navi::NAVI_INVALID_CAST_OFFSET;                       \
        } else {                                                               \
            auto offsetObj = (((char *)typedObj) - ((char *)obj));             \
            navi::updateCastOffset(offsetObj, *objOffset);                     \
        }                                                                      \
    }

#define CAST_OFFSET(member)                                                    \
    {                                                                          \
        if (unlikely(objOffset && dependOffset)) {                             \
            CAST_OFFSET_OBJ();                                                 \
            if (navi::NAVI_INVALID_CAST_OFFSET != *dependOffset) {             \
                auto *typedDepend =                                            \
                    dynamic_cast<decltype(typedObj->member)>(depend);          \
                if (!typedDepend) {                                            \
                    *dependOffset = navi::NAVI_INVALID_CAST_OFFSET;            \
                } else {                                                       \
                    auto offsetDepend =                                        \
                        (((char *)typedDepend) - ((char *)depend));            \
                    navi::updateCastOffset(offsetDepend, *dependOffset);       \
                }                                                              \
            }                                                                  \
            return false;                                                      \
        }                                                                      \
    }

#define DYNAMIC_CAST_OFFSET(member)                                            \
    {                                                                          \
        if (unlikely(objOffset && dependOffset)) {                             \
            using T =                                                          \
                std::remove_const_t<std::remove_pointer_t<decltype(this)>>;    \
            CAST_OFFSET_OBJ();                                                 \
            using _MemberElemT = std::remove_pointer_t<                        \
                typename std::decay_t<decltype(std::declval<T>().member)>      \
                    ::value_type>;                                              \
            navi::dynamicResourceCastOffset<_MemberElemT>(depend, dependOffset);     \
            return false;                                                      \
        }                                                                      \
    }

#define BIND_RESOURCE_TO(member)                                               \
    [](void *obj,                                                              \
       void *castObj,                                                          \
       navi::Resource *depend,                                                 \
       int64_t *objOffset,                                                     \
       int64_t *dependOffset) -> bool {                                        \
        using T = std::remove_const_t<std::remove_pointer_t<decltype(this)>>;  \
        CAST_OFFSET(member);                                                   \
        T *typedObj;                                                           \
        if (castObj) {                                                         \
            typedObj = (T *)castObj;                                           \
            assert(typedObj == navi_bind_cast<T>(obj));                        \
        } else {                                                               \
            typedObj = navi_bind_cast<T>(obj);                                 \
        }                                                                      \
        if (unlikely(!typedObj)) {                                             \
            NAVI_KERNEL_LOG(ERROR,                                             \
                            "bind resource failed, cast obj to class [%s] "    \
                            "failed, obj [%p], "                               \
                            "member: [%s] [%s], file: [%s:%d]",                \
                            typeid(T).name(),                                  \
                            obj,                                               \
                            typeid(decltype(typedObj->member)).name(),         \
                            #member,                                           \
                            __FILE__,                                          \
                            __LINE__);                                         \
            return false;                                                      \
        }                                                                      \
        typedObj->member = nullptr;                                            \
        if (!depend) {                                                         \
            return true;                                                       \
        }                                                                      \
        using DependType = std::remove_const_t<                                \
            std::remove_pointer_t<decltype(typedObj->member)>>;                \
        DependType *typedDepend;                                               \
        auto castDepend = depend->getCastPtr();                                \
        if (castDepend) {                                                      \
            typedDepend = (DependType *)castDepend;                            \
            assert(typedDepend == dynamic_cast<DependType *>(depend));         \
        } else {                                                               \
            typedDepend = dynamic_cast<DependType *>(depend);                  \
        }                                                                      \
        typedObj->member = typedDepend;                                        \
        auto ret = (typedDepend != nullptr);                                   \
        if (!ret) {                                                            \
            NAVI_KERNEL_LOG(                                                   \
                ERROR,                                                         \
                "bind resource failed, can't cast type [%s] to type [%s], "    \
                "class: [%s], member: [%s], file: [%s:%d]",                    \
                typeid(*depend).name(),                                        \
                typeid(decltype(typedObj->member)).name(),                     \
                typeid(T).name(),                                              \
                #member,                                                       \
                __FILE__,                                                      \
                __LINE__);                                                     \
        }                                                                      \
        return ret;                                                            \
    }

#define BIND_DYNAMIC_RESOURCE_TO(member)                                       \
    [](void *obj,                                                              \
       void *castObj,                                                          \
       navi::Resource *depend,                                                 \
       int64_t *objOffset,                                                     \
       int64_t *dependOffset,                                                  \
       void *not_used) -> bool {                                               \
        DYNAMIC_CAST_OFFSET(member);                                           \
        if (!depend) {                                                         \
            return true;                                                       \
        }                                                                      \
        using T = std::remove_const_t<std::remove_pointer_t<decltype(this)>>;  \
        T *typedObj = navi_bind_cast<T>(obj);                                  \
        const char *typeName = nullptr;                                        \
        auto ret = bindDynamicResourceTo(depend, typedObj->member, typeName);  \
        if (!ret) {                                                            \
            NAVI_KERNEL_LOG(ERROR,                                             \
                            "bind dynamic resource failed, can't cast type "   \
                            "[%s] to type [%s], class: [%s], set member: "     \
                            "[%s], file: [%s:%d]",                             \
                            typeid(*depend).name(),                            \
                            typeName,                                          \
                            typeid(T).name(),                                  \
                            #member,                                           \
                            __FILE__,                                          \
                            __LINE__);                                         \
        }                                                                      \
        return ret;                                                            \
    }
}

#define RESOURCE_DEPEND_DECLARE()                                                                                      \
public:                                                                                                                \
    __attribute__((always_inline)) void __navi__declare_depend(navi::ResourceDefBuilder &builder) override {           \
        navi::DependMatcher<512> *matcherEnd = nullptr;                                                                \
        __navi__declare_depend_t(matcherEnd, builder);                                                                 \
    }                                                                                                                  \
                                                                                                                       \
protected:                                                                                                             \
    template <int N>                                                                                                   \
    __attribute__((always_inline)) void __navi__declare_depend_t(navi::DependMatcher<N> *matcher,                      \
                                                                 navi::ResourceDefBuilder &builder) {                  \
        __navi__declare_depend_t((navi::DependMatcher<N - 1> *)0, builder);                                            \
    }                                                                                                                  \
    using navi::Resource::__navi__declare_depend_t;

#define RESOURCE_DEPEND_DECLARE_BASE(BASE)                                                                             \
public:                                                                                                                \
    __attribute__((always_inline)) void __navi__declare_depend(navi::ResourceDefBuilder &builder) override {           \
        BASE::__navi__declare_depend(builder);                                                                         \
        navi::DependMatcher<512> *matcherEnd = nullptr;                                                                \
        __navi__declare_depend_t(matcherEnd, builder);                                                                 \
    }                                                                                                                  \
                                                                                                                       \
protected:                                                                                                             \
    template <int N>                                                                                                   \
    __attribute__((always_inline)) void __navi__declare_depend_t(navi::DependMatcher<N> *matcher,                      \
                                                                 navi::ResourceDefBuilder &builder) {                  \
        __navi__declare_depend_t((navi::DependMatcher<N - 1> *)0, builder);                                            \
    }                                                                                                                  \
    using navi::Resource::__navi__declare_depend_t;

#define RESOURCE_DEPEND_ON(CLASS, NAME) RESOURCE_DEPEND_ON_ID(CLASS, NAME, CLASS::RESOURCE_ID, true)
#define RESOURCE_DEPEND_ON_FALSE(CLASS, NAME) RESOURCE_DEPEND_ON_ID(CLASS, NAME, CLASS::RESOURCE_ID, false)

#define RESOURCE_DEPEND_ON_ID(CLASS, NAME, ID, REQUIRE)                                                                \
    RESOURCE_DECLARE_DEPEND_ON_HELPER(__COUNTER__, CLASS, NAME, ID, REQUIRE)
#define RESOURCE_DECLARE_DEPEND_ON_HELPER(ctr, CLASS, NAME, ID, REQUIRE)                                               \
    RESOURCE_DECLARE_DEPEND_ON_UNIQ(ctr, CLASS, NAME, ID, REQUIRE)
#define RESOURCE_DECLARE_DEPEND_ON_UNIQ(ctr, CLASS, NAME, ID, REQUIRE)         \
    union {                                                                    \
        CLASS *const NAME = nullptr;                                           \
        CLASS *NAME##__;                                                       \
    };                                                                         \
    template <typename T>                                                      \
    static bool __navi_resource_binder_func_##NAME(void *obj,                  \
                                                   void *castObj,              \
                                                   navi::Resource *depend,     \
                                                   int64_t *objOffset,         \
                                                   int64_t *dependOffset) {    \
        CAST_OFFSET(NAME##__);                                                 \
        T *typedObj;                                                           \
        if (castObj) {                                                         \
            typedObj = (T *)castObj;                                           \
            assert(typedObj == navi_bind_cast<T>(obj));                        \
        } else {                                                               \
            typedObj = navi_bind_cast<T>(obj);                                 \
        }                                                                      \
        typedObj->NAME##__ = nullptr;                                          \
        if (!depend) {                                                         \
            return true;                                                       \
        }                                                                      \
        using DependType = std::remove_const_t<                                \
            std::remove_pointer_t<decltype(typedObj->NAME##__)>>;              \
        DependType *typedDepend;                                               \
        auto castDepend = depend->getCastPtr();                                \
        if (castDepend) {                                                      \
            typedDepend = (DependType *)castDepend;                            \
            assert(typedDepend == dynamic_cast<DependType *>(depend));         \
        } else {                                                               \
            typedDepend = dynamic_cast<DependType *>(depend);                  \
        }                                                                      \
        typedObj->NAME##__ = typedDepend;                                      \
        auto ret = (typedDepend != nullptr);                                   \
        if (!ret) {                                                            \
            NAVI_KERNEL_LOG(                                                   \
                ERROR,                                                         \
                "bind resource failed, can't cast type [%s] to type [%s], "    \
                "class: [%s], member: [%s], file: [%s:%d]",                    \
                typeid(*depend).name(),                                        \
                typeid(decltype(typedObj->NAME##__)).name(),                   \
                typeid(T).name(),                                              \
                #NAME,                                                         \
                __FILE__,                                                      \
                __LINE__);                                                     \
        }                                                                      \
        return ret;                                                            \
    }                                                                          \
    __attribute__((always_inline)) void __navi__declare_depend_t(              \
        navi::DependMatcher<ctr> *matcher,                                     \
        navi::ResourceDefBuilder &builder) {                                   \
        builder.dependOn(                                                      \
            ID,                                                                \
            REQUIRE,                                                           \
            __navi_resource_binder_func_##NAME<                                \
                std::remove_const_t<std::remove_pointer_t<decltype(this)>>>);  \
        navi::DependMatcher<ctr - 1> *nextMatcher = nullptr;                   \
        __navi__declare_depend_t(nextMatcher, builder);                        \
    }

#define RESOURCE_NAMED_DATA(CLASS, NAME, ID) RESOURCE_DECLARE_NAMED_DATA_HELPER(__COUNTER__, CLASS, NAME, ID, true)
#define RESOURCE_NAMED_DATA_FALSE(CLASS, NAME, ID)                                                                     \
    RESOURCE_DECLARE_NAMED_DATA_HELPER(__COUNTER__, CLASS, NAME, ID, false)

#define RESOURCE_DECLARE_NAMED_DATA_HELPER(ctr, CLASS, NAME, ID, REQUIRE)                                              \
    RESOURCE_DECLARE_NAMED_DATA_UNIQ(ctr, CLASS, NAME, ID, REQUIRE)
#define RESOURCE_DECLARE_NAMED_DATA_UNIQ(ctr, CLASS, NAME, ID, REQUIRE)        \
    union {                                                                    \
        CLASS *const NAME = nullptr;                                           \
        CLASS *NAME##__;                                                       \
    };                                                                         \
    template <typename T>                                                      \
    static bool __navi_named_data_binder_func_##NAME(                          \
        void *obj, void *castObj, navi::Data *namedData, int64_t *objOffset) { \
        if (unlikely(nullptr != objOffset)) {                                  \
            CAST_OFFSET_OBJ();                                                 \
            return false;                                                      \
        }                                                                      \
        T *typedObj;                                                           \
        if (castObj) {                                                         \
            typedObj = (T *)castObj;                                           \
            assert(typedObj == navi_bind_cast<T>(obj));                        \
        } else {                                                               \
            typedObj = navi_bind_cast<T>(obj);                                 \
        }                                                                      \
        typedObj->NAME##__ = nullptr;                                          \
        if (!namedData) {                                                      \
            return true;                                                       \
        }                                                                      \
        auto *typedData =                                                      \
            dynamic_cast<decltype(typedObj->NAME##__)>(namedData);             \
        typedObj->NAME##__ = typedData;                                        \
        auto ret = (typedData != nullptr);                                     \
        if (!ret) {                                                            \
            NAVI_KERNEL_LOG(                                                   \
                ERROR,                                                         \
                "bind named data failed, can't cast type [%s] to type [%s], "  \
                "class: [%s], member: [%s], file: [%s:%d]",                    \
                typeid(*namedData).name(),                                     \
                typeid(decltype(typedObj->NAME##__)).name(),                   \
                typeid(T).name(),                                              \
                #NAME,                                                         \
                __FILE__,                                                      \
                __LINE__);                                                     \
        }                                                                      \
        return ret;                                                            \
    }                                                                          \
    __attribute__((always_inline)) void __navi__declare_depend_t(              \
        navi::DependMatcher<ctr> *matcher,                                     \
        navi::ResourceDefBuilder &builder) {                                   \
        builder.namedData(                                                     \
            ID,                                                                \
            REQUIRE,                                                           \
            __navi_named_data_binder_func_##NAME<                              \
                std::remove_const_t<std::remove_pointer_t<decltype(this)>>>);  \
        navi::DependMatcher<ctr - 1> *nextMatcher = nullptr;                   \
        __navi__declare_depend_t(nextMatcher, builder);                        \
    }
