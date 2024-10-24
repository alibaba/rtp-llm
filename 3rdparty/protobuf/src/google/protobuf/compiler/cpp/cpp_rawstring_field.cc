// Protocol Buffers - Google's data interchange format
// Copyright 2008 Google Inc.  All rights reserved.
// https://developers.google.com/protocol-buffers/
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//     * Neither the name of Google Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

// Author: kenton@google.com (Kenton Varda)
//  Based on original Protocol Buffers design by
//  Sanjay Ghemawat, Jeff Dean, and others.

#include <google/protobuf/compiler/cpp/cpp_rawstring_field.h>
#include <google/protobuf/compiler/cpp/cpp_helpers.h>
#include <google/protobuf/descriptor.pb.h>
#include <google/protobuf/io/printer.h>
#include <google/protobuf/rawstring.h>
#include <google/protobuf/stubs/strutil.h>

namespace google {
namespace protobuf {
namespace compiler {
namespace cpp {

namespace {

void SetRawStringVariables(const FieldDescriptor* descriptor,
                           std::map<std::string, std::string>* variables,
                           const Options& options) {
  SetCommonFieldVariables(descriptor, variables, options);
  (*variables)["declared_type"] = "RawString"; // replaced declared type
  (*variables)["default"] = DefaultValue(options, descriptor);
  (*variables)["default_length"] =
      StrCat(descriptor->default_value_rawstring().size());
  std::string default_variable_string = MakeDefaultName(descriptor);
  (*variables)["default_variable_name"] = default_variable_string;
  (*variables)["default_variable"] =
      descriptor->default_value_rawstring().empty()
          ? "&::" + (*variables)["proto_ns"] +
                "::internal::GetEmptyRawStringAlreadyInited()"
          : "&" + QualifiedClassName(descriptor->containing_type(), options) +
                "::" + default_variable_string + ".get()";
  (*variables)["pointer_type"] = 
      descriptor->type() == FieldDescriptor::TYPE_BYTES ? "void" : "char";
  (*variables)["null_check"] = (*variables)["DCHK"] + "(value != nullptr);\n";
  // NOTE: Escaped here to unblock proto1->proto2 migration.
  // TODO(liujisi): Extend this to apply for other conflicting methods.
  (*variables)["release_name"] =
      SafeFunctionName(descriptor->containing_type(), descriptor, "release_");
  (*variables)["full_name"] = descriptor->full_name();

  (*variables)["lite"] =
      HasDescriptorMethods(descriptor->file(), options) ? "" : "Lite";
}

}  // namespace

// ===================================================================

RawStringFieldGenerator::RawStringFieldGenerator(const FieldDescriptor* descriptor,
                                           const Options& options)
    : FieldGenerator(descriptor, options),
      lite_(!HasDescriptorMethods(descriptor->file(), options)),
      inlined_(IsStringInlined(descriptor, options)) {
  inlined_ = false; // TODO: support on InlineRawStringField, opensource code inline always false
  SetRawStringVariables(descriptor, &variables_, options);
}

RawStringFieldGenerator::~RawStringFieldGenerator() {}

void RawStringFieldGenerator::GeneratePrivateMembers(io::Printer* printer) const {
  Formatter format(printer, variables_);
  if (inlined_) {
    format("::$proto_ns$::internal::InlinedStringField $name$_;\n");
  } else {
    // N.B. that we continue to use |ArenaRawStringPtr| instead of |RawString*| for
    // RawString fields, even when SupportArenas(descriptor_) == false. Why?  The
    // simple answer is to avoid unmaintainable complexity. The reflection code
    // assumes ArenaRawStringPtrs. These are *almost* in-memory-compatible with
    // string*, except for the pointer tags and related ownership semantics. We
    // could modify the runtime code to use RawString* for the
    // not-supporting-arenas case, but this would require a way to detect which
    // type of class was generated (adding overhead and complexity to
    // GeneratedMessageReflection) and littering the runtime code paths with
    // conditionals. It's simpler to stick with this but use lightweight
    // accessors that assume arena == NULL.  There should be very little
    // overhead anyway because it's just a tagged pointer in-memory.
    format("::$proto_ns$::internal::ArenaRawStringPtr $name$_;\n");
  }
}

void RawStringFieldGenerator::GenerateStaticMembers(io::Printer* printer) const {
  Formatter format(printer, variables_);
  if (!descriptor_->default_value_rawstring().empty()) {
    // We make the default instance public, so it can be initialized by
    // non-friend code.
    format(
        "public:\n"
        "static ::$proto_ns$::internal::ExplicitlyConstructed<std::RawString>"
        " $default_variable_name$;\n"
        "private:\n");
  }
}

void RawStringFieldGenerator::GenerateAccessorDeclarations(
    io::Printer* printer) const {
  Formatter format(printer, variables_);

  format(
      "$deprecated_attr$const std::RawString& ${1$$name$$}$() const;\n",
      descriptor_);

  format(
      "$deprecated_attr$void ${1$set_$name$$}$(const std::string& value);\n"
      "$deprecated_attr$void ${1$set_$name$$}$(std::string&& value);\n"
      "$deprecated_attr$void ${1$set_$name$$}$(const char* value);\n"
      "$deprecated_attr$void ${1$set_$name$$}$(const $pointer_type$* "
      "value, size_t size)"
      ";\n"
      "$deprecated_attr$void ${1$set_$name$$}$(const std::shared_ptr<const char>& "
      "value, size_t size);\n"
      "$deprecated_attr$void ${1$set_$name$$}$(const std::RawString& value);\n",
      descriptor_);
  
  format(
      "$deprecated_attr$std::RawString* ${1$mutable_$name$$}$();\n"
      "$deprecated_attr$std::RawString* ${1$$release_name$$}$();\n"
      "$deprecated_attr$void ${1$set_allocated_$name$$}$(std::RawString* "
      "$name$);\n",
      descriptor_);

  if (SupportsArenas(descriptor_)) {
    format(
        "$GOOGLE_PROTOBUF$_RUNTIME_DEPRECATED(\"The unsafe_arena_ accessors "
        "for\"\n"
        "\"    rawstring fields are deprecated and will be removed in a\"\n"
        "\"    future release.\")\n"
        "std::RawString* ${1$unsafe_arena_release_$name$$}$();\n"
        "$GOOGLE_PROTOBUF$_RUNTIME_DEPRECATED(\"The unsafe_arena_ accessors "
        "for\"\n"
        "\"    rawstring fields are deprecated and will be removed in a\"\n"
        "\"    future release.\")\n"
        "void ${1$unsafe_arena_set_allocated_$name$$}$(\n"
        "    std::RawString* $name$);\n",
        descriptor_);
  }
}

void RawStringFieldGenerator::GenerateInlineAccessorDefinitions(
    io::Printer* printer) const {
  Formatter format(printer, variables_);
  if (SupportsArenas(descriptor_)) {
    format(
        "inline const std::RawString& $classname$::$name$() const {\n"
        "  // @@protoc_insertion_point(field_get:$full_name$)\n"
        "  return $name$_.Get();\n"
        "}\n"
    );

    format(
       "inline void $classname$::set_$name$(const std::string& value) {\n"
        "  $set_hasbit$\n"
        "  $name$_.Set$lite$($default_variable$, std::RawString(value), GetArenaNoVirtual());\n"
        "  // @@protoc_insertion_point(field_set:$full_name$)\n"
        "}\n"
        "inline void $classname$::set_$name$(std::string&& value) {\n"
        "  $set_hasbit$\n"
        "  $name$_.Set$lite$(\n"
        "    $default_variable$, std::RawString(::std::move(value)), GetArenaNoVirtual());\n"
        "  // @@protoc_insertion_point(field_set_rvalue:$full_name$)\n"
        "}\n"
        "inline void $classname$::set_$name$(const char* value) {\n"
        "  $null_check$"
        "  $set_hasbit$\n"
        "  $name$_.Set$lite$($default_variable$, std::RawString(value),\n"
        "              GetArenaNoVirtual());\n"
        "  // @@protoc_insertion_point(field_set_char:$full_name$)\n"
        "}\n"
        "inline "
        "void $classname$::set_$name$(const $pointer_type$* value,\n"
        "    size_t size) {\n"
        "  $set_hasbit$\n"
        "  $name$_.Set$lite$($default_variable$, std::RawString(\n"
        "      reinterpret_cast<const char*>(value), size), "
        "GetArenaNoVirtual());\n"
        "  // @@protoc_insertion_point(field_set_pointer:$full_name$)\n"
        "}\n"
        "inline "
        "void $classname$::set_$name$(const std::shared_ptr<const char>& value,\n"
        "    size_t size) {\n"
        "  $set_hasbit$\n"
        "  $name$_.Set$lite$($default_variable$, std::RawString(value, size), "
        "GetArenaNoVirtual());\n"
        "  // @@protoc_insertion_point(field_set_pointer:$full_name$)\n"
        "}\n"
        "inline void $classname$::set_$name$(const std::RawString& value) {\n"
        "  $set_hasbit$\n"
        "  $name$_.Set$lite$($default_variable$, value, GetArenaNoVirtual());\n"
        "  // @@protoc_insertion_point(field_set:$full_name$)\n"
        "}\n"
    );

    format(
        "inline std::RawString* $classname$::mutable_$name$() {\n"
        "  $set_hasbit$\n"
        "  // @@protoc_insertion_point(field_mutable:$full_name$)\n"
        "  return $name$_.Mutable($default_variable$, GetArenaNoVirtual());\n"
        "}\n"
        "inline std::RawString* $classname$::$release_name$() {\n"
        "  // @@protoc_insertion_point(field_release:$full_name$)\n");

    if (HasFieldPresence(descriptor_->file())) {
      format(
          "  if (!has_$name$()) {\n"
          "    return nullptr;\n"
          "  }\n"
          "  $clear_hasbit$\n"
          "  return $name$_.ReleaseNonDefault("
          "$default_variable$, GetArenaNoVirtual());\n");
    } else {
      format(
          "  $clear_hasbit$\n"
          "  return $name$_.Release($default_variable$, "
          "GetArenaNoVirtual());\n");
    }

    format(
        "}\n"
        "inline void $classname$::set_allocated_$name$(std::RawString* $name$) {\n"
        "  if ($name$ != nullptr) {\n"
        "    $set_hasbit$\n"
        "  } else {\n"
        "    $clear_hasbit$\n"
        "  }\n"
        "  $name$_.SetAllocated($default_variable$, $name$,\n"
        "      GetArenaNoVirtual());\n"
        "  // @@protoc_insertion_point(field_set_allocated:$full_name$)\n"
        "}\n");

    format(
        "inline std::RawString* $classname$::unsafe_arena_release_$name$() {\n"
        "  // "
        "@@protoc_insertion_point(field_unsafe_arena_release:$full_name$)\n"
        "  $DCHK$(GetArenaNoVirtual() != nullptr);\n"
        "  $clear_hasbit$\n"
        "  return $name$_.UnsafeArenaRelease($default_variable$,\n"
        "      GetArenaNoVirtual());\n"
        "}\n"
        "inline void $classname$::unsafe_arena_set_allocated_$name$(\n"
        "    std::RawString* $name$) {\n"
        "  $DCHK$(GetArenaNoVirtual() != nullptr);\n"
        "  if ($name$ != nullptr) {\n"
        "    $set_hasbit$\n"
        "  } else {\n"
        "    $clear_hasbit$\n"
        "  }\n"
        "  $name$_.UnsafeArenaSetAllocated($default_variable$,\n"
        "      $name$, GetArenaNoVirtual());\n"
        "  // @@protoc_insertion_point(field_unsafe_arena_set_allocated:"
        "$full_name$)\n"
        "}\n");
  } else {
    // No-arena case.
    format(
        "inline const std::RawString& $classname$::$name$() const {\n"
        "  // @@protoc_insertion_point(field_get:$full_name$)\n"
        "  return $name$_.GetNoArena();\n"
        "}\n"
    );

    format(
        "inline void $classname$::set_$name$(const std::string& value) {\n"
        "  $set_hasbit$\n"
        "  $name$_.SetNoArena($default_variable$, std::RawString(value));\n"
        "  // @@protoc_insertion_point(field_set:$full_name$)\n"
        "}\n"
        "inline void $classname$::set_$name$(std::string&& value) {\n"
        "  $set_hasbit$\n"
        "  $name$_.SetNoArena(\n"
        "    $default_variable$, std::RawString(::std::move(value)));\n"
        "  // @@protoc_insertion_point(field_set_rvalue:$full_name$)\n"
        "}\n"
        "inline void $classname$::set_$name$(const char* value) {\n"
        "  $null_check$"
        "  $set_hasbit$\n"
        "  $name$_.SetNoArena($default_variable$, std::RawString(value));\n"
        "  // @@protoc_insertion_point(field_set_char:$full_name$)\n"
        "}\n"
        "inline "
        "void $classname$::set_$name$(const $pointer_type$* value, "
        "size_t size) {\n"
        "  $set_hasbit$\n"
        "  $name$_.SetNoArena($default_variable$,\n"
        "      std::RawString(reinterpret_cast<const char*>(value), size));\n"
        "  // @@protoc_insertion_point(field_set_pointer:$full_name$)\n"
        "}\n"
        "inline "
        "void $classname$::set_$name$(const std::shared_ptr<const char>& value, "
        "size_t size) {\n"
        "  $set_hasbit$\n"
        "  $name$_.SetNoArena($default_variable$, std::RawString(value, size));\n"
        "  // @@protoc_insertion_point(field_set_pointer:$full_name$)\n"
        "}\n"
        "inline void $classname$::set_$name$(const std::RawString& value) {\n"
        "  $set_hasbit$\n"
        "  $name$_.SetNoArena($default_variable$, value);\n"
        "  // @@protoc_insertion_point(field_set_char:$full_name$)\n"
        "}\n"
    );
    
    format(
        "inline std::RawString* $classname$::mutable_$name$() {\n"
        "  $set_hasbit$\n"
        "  // @@protoc_insertion_point(field_mutable:$full_name$)\n"
        "  return $name$_.MutableNoArena($default_variable$);\n"
        "}\n"
        "inline std::RawString* $classname$::$release_name$() {\n"
        "  // @@protoc_insertion_point(field_release:$full_name$)\n");

    if (HasFieldPresence(descriptor_->file())) {
      format(
          "  if (!has_$name$()) {\n"
          "    return nullptr;\n"
          "  }\n"
          "  $clear_hasbit$\n"
          "  return $name$_.ReleaseNonDefaultNoArena($default_variable$);\n");
    } else {
      format(
          "  $clear_hasbit$\n"
          "  return $name$_.ReleaseNoArena($default_variable$);\n");
    }

    format(
        "}\n"
        "inline void $classname$::set_allocated_$name$(std::RawString* $name$) {\n"
        "  if ($name$ != nullptr) {\n"
        "    $set_hasbit$\n"
        "  } else {\n"
        "    $clear_hasbit$\n"
        "  }\n"
        "  $name$_.SetAllocatedNoArena($default_variable$, $name$);\n"
        "  // @@protoc_insertion_point(field_set_allocated:$full_name$)\n"
        "}\n");
  }
}

void RawStringFieldGenerator::GenerateNonInlineAccessorDefinitions(
    io::Printer* printer) const {
  Formatter format(printer, variables_);
  if (!descriptor_->default_value_rawstring().empty()) {
    // Initialized in GenerateDefaultInstanceAllocator.
    format(
        "::$proto_ns$::internal::ExplicitlyConstructed<std::RawString> "
        "$classname$::$default_variable_name$;\n");
  }
}

void RawStringFieldGenerator::GenerateClearingCode(io::Printer* printer) const {
  Formatter format(printer, variables_);
  // Two-dimension specialization here: supporting arenas or not, and default
  // value is the empty string or not. Complexity here ensures the minimal
  // number of branches / amount of extraneous code at runtime (given that the
  // below methods are inlined one-liners)!
  if (SupportsArenas(descriptor_)) {
    if (descriptor_->default_value_rawstring().empty()) {
      format(
          "$name$_.ClearToEmpty($default_variable$, GetArenaNoVirtual());\n");
    } else {
      format(
          "$name$_.ClearToDefault($default_variable$, GetArenaNoVirtual());\n");
    }
  } else {
    if (descriptor_->default_value_rawstring().empty()) {
      format("$name$_.ClearToEmptyNoArena($default_variable$);\n");
    } else {
      format("$name$_.ClearToDefaultNoArena($default_variable$);\n");
    }
  }
}

void RawStringFieldGenerator::GenerateMessageClearingCode(
    io::Printer* printer) const {
  Formatter format(printer, variables_);
  // Two-dimension specialization here: supporting arenas, field presence, or
  // not, and default value is the empty string or not. Complexity here ensures
  // the minimal number of branches / amount of extraneous code at runtime
  // (given that the below methods are inlined one-liners)!

  // If we have field presence, then the Clear() method of the protocol buffer
  // will have checked that this field is set.  If so, we can avoid redundant
  // checks against default_variable.
  const bool must_be_present = HasFieldPresence(descriptor_->file());

  if (inlined_ && must_be_present) {
    // Calling mutable_$name$() gives us a rawstring reference and sets the has bit
    // for $name$ (in proto2).  We may get here when the string field is inlined
    // but the string's contents have not been changed by the user, so we cannot
    // make an assertion about the contents of the string and could never make
    // an assertion about the string instance.
    //
    // For non-inlined strings, we distinguish from non-default by comparing
    // instances, rather than contents.
    format("$DCHK$(!$name$_.IsDefault($default_variable$));\n");
  }

  if (SupportsArenas(descriptor_)) {
    if (descriptor_->default_value_rawstring().empty()) {
      if (must_be_present) {
        format("$name$_.ClearNonDefaultToEmpty();\n");
      } else {
        format(
            "$name$_.ClearToEmpty($default_variable$, GetArenaNoVirtual());\n");
      }
    } else {
      // Clear to a non-empty default is more involved, as we try to use the
      // Arena if one is present and may need to reallocate the string.
      format(
          "$name$_.ClearToDefault($default_variable$, GetArenaNoVirtual());\n");
    }
  } else if (must_be_present) {
    // When Arenas are disabled and field presence has been checked, we can
    // safely treat the ArenaStringPtr as a string*.
    if (descriptor_->default_value_rawstring().empty()) {
      format("$name$_.ClearNonDefaultToEmptyNoArena();\n");
    } else {
      format("$name$_.UnsafeMutablePointer()->assign(*$default_variable$);\n");
    }
  } else {
    if (descriptor_->default_value_rawstring().empty()) {
      format("$name$_.ClearToEmptyNoArena($default_variable$);\n");
    } else {
      format("$name$_.ClearToDefaultNoArena($default_variable$);\n");
    }
  }
}

void RawStringFieldGenerator::GenerateMergingCode(io::Printer* printer) const {
  Formatter format(printer, variables_);
  if (SupportsArenas(descriptor_) || descriptor_->containing_oneof() != NULL) {
    // TODO(gpike): improve this
    format("set_$name$(from.$name$());\n");
  } else {
    format(
        "$set_hasbit$\n"
        "$name$_.AssignWithDefault($default_variable$, from.$name$_);\n");
  }
}

void RawStringFieldGenerator::GenerateSwappingCode(io::Printer* printer) const {
  Formatter format(printer, variables_);
  if (inlined_) {
    format("$name$_.Swap(&other->$name$_);\n");
  } else {
    format(
        "$name$_.Swap(&other->$name$_, $default_variable$,\n"
        "  GetArenaNoVirtual());\n");
  }
}

void RawStringFieldGenerator::GenerateConstructorCode(io::Printer* printer) const {
  Formatter format(printer, variables_);
  // TODO(ckennelly): Construct non-empty strings as part of the initializer
  // list.
  if (inlined_ && descriptor_->default_value_rawstring().empty()) {
    // Automatic initialization will construct the string.
    return;
  }

  format("$name$_.UnsafeSetDefault($default_variable$);\n");
}

void RawStringFieldGenerator::GenerateCopyConstructorCode(
    io::Printer* printer) const {
  Formatter format(printer, variables_);
  GenerateConstructorCode(printer);

  if (HasFieldPresence(descriptor_->file())) {
    format("if (from.has_$name$()) {\n");
  } else {
    format("if (from.$name$().size() > 0) {\n");
  }

  format.Indent();

  if (SupportsArenas(descriptor_) || descriptor_->containing_oneof() != NULL) {
    // TODO(gpike): improve this
    format(
        "$name$_.Set$lite$($default_variable$, from.$name$(),\n"
        "  GetArenaNoVirtual());\n");
  } else {
    format("$name$_.AssignWithDefault($default_variable$, from.$name$_);\n");
  }

  format.Outdent();
  format("}\n");
}

void RawStringFieldGenerator::GenerateDestructorCode(io::Printer* printer) const {
  Formatter format(printer, variables_);
  if (inlined_) {
    // The destructor is automatically invoked.
    return;
  }

  format("$name$_.DestroyNoArena($default_variable$);\n");
}

bool RawStringFieldGenerator::GenerateArenaDestructorCode(
    io::Printer* printer) const {
  if (!inlined_) {
    return false;
  }

  Formatter format(printer, variables_);
  format("_this->$name$_.DestroyNoArena($default_variable$);\n");
  return true;
}

void RawStringFieldGenerator::GenerateDefaultInstanceAllocator(
    io::Printer* printer) const {
  Formatter format(printer, variables_);
  if (!descriptor_->default_value_rawstring().empty()) {
    format(
        "$ns$::$classname$::$default_variable_name$.DefaultConstruct();\n"
        "*$ns$::$classname$::$default_variable_name$.get_mutable() = "
        "std::RawString($default$, $default_length$);\n"
        "::$proto_ns$::internal::OnShutdownDestroyRawString(\n"
        "    $ns$::$classname$::$default_variable_name$.get_mutable());\n");
  }
}

void RawStringFieldGenerator::GenerateMergeFromCodedStream(
    io::Printer* printer) const {
  Formatter format(printer, variables_);
  format(
    "DO_(::$proto_ns$::internal::WireFormatLite::Read$declared_type$(\n"
    "      input, this->mutable_$name$()));\n");
}

bool RawStringFieldGenerator::MergeFromCodedStreamNeedsArena() const {
  return false;
}

void RawStringFieldGenerator::GenerateSerializeWithCachedSizes(
    io::Printer* printer) const {
  Formatter format(printer, variables_);
  format(
      "::$proto_ns$::internal::WireFormatLite::Write$declared_type$"
      "(\n"
      "  $number$, this->$name$(), output);\n");
}

void RawStringFieldGenerator::GenerateSerializeWithCachedSizesToArray(
    io::Printer* printer) const {
  Formatter format(printer, variables_);
  format(
      "target =\n"
      "  ::$proto_ns$::internal::WireFormatLite::Write$declared_type$ToArray(\n"
      "    $number$, this->$name$(), target);\n");
}

void RawStringFieldGenerator::GenerateByteSize(io::Printer* printer) const {
  Formatter format(printer, variables_);
  format(
      "total_size += $tag_size$ +\n"
      "  ::$proto_ns$::internal::WireFormatLite::$declared_type$Size(\n"
      "    this->$name$());\n");
}

uint32 RawStringFieldGenerator::CalculateFieldTag() const {
  return inlined_ ? 1 : 0;
}

// ===================================================================

RawStringOneofFieldGenerator::RawStringOneofFieldGenerator(
    const FieldDescriptor* descriptor, const Options& options)
    : RawStringFieldGenerator(descriptor, options) {
  inlined_ = false;
  SetCommonOneofFieldVariables(descriptor, &variables_);
}

RawStringOneofFieldGenerator::~RawStringOneofFieldGenerator() {}

void RawStringOneofFieldGenerator::GenerateInlineAccessorDefinitions(
    io::Printer* printer) const {
  Formatter format(printer, variables_);
  if (SupportsArenas(descriptor_)) {
    format(
        "inline const std::RawString& $classname$::$name$() const {\n"
        "  // @@protoc_insertion_point(field_get:$full_name$)\n"
        "  if (has_$name$()) {\n"
        "    return $field_member$.Get();\n"
        "  }\n"
        "  return *$default_variable$;\n"
        "}\n"
    );

    format(
       "inline void $classname$::set_$name$(const std::string& value) {\n"
        "  if (!has_$name$()) {\n"
        "    clear_$oneof_name$();\n"
        "    set_has_$name$();\n"
        "    $field_member$.UnsafeSetDefault($default_variable$);\n"
        "  }\n"
        "  $field_member$.Set$lite$($default_variable$, std::RawString(value),\n"
        "      GetArenaNoVirtual());\n"
        "  // @@protoc_insertion_point(field_set:$full_name$)\n"
        "}\n"
        "inline void $classname$::set_$name$(std::string&& value) {\n"
        "  // @@protoc_insertion_point(field_set:$full_name$)\n"
        "  if (!has_$name$()) {\n"
        "    clear_$oneof_name$();\n"
        "    set_has_$name$();\n"
        "    $field_member$.UnsafeSetDefault($default_variable$);\n"
        "  }\n"
        "  $field_member$.Set$lite$(\n"
        "    $default_variable$, std::RawString(::std::move(value)), GetArenaNoVirtual());\n"
        "  // @@protoc_insertion_point(field_set_rvalue:$full_name$)\n"
        "}\n"
        "inline void $classname$::set_$name$(const char* value) {\n"
        "  $null_check$"
        "  if (!has_$name$()) {\n"
        "    clear_$oneof_name$();\n"
        "    set_has_$name$();\n"
        "    $field_member$.UnsafeSetDefault($default_variable$);\n"
        "  }\n"
        "  $field_member$.Set$lite$($default_variable$,\n"
        "      std::RawString(value), GetArenaNoVirtual());\n"
        "  // @@protoc_insertion_point(field_set_char:$full_name$)\n"
        "}\n"
        "inline "
        "void $classname$::set_$name$(const $pointer_type$* value,\n"
        "                             size_t size) {\n"
        "  if (!has_$name$()) {\n"
        "    clear_$oneof_name$();\n"
        "    set_has_$name$();\n"
        "    $field_member$.UnsafeSetDefault($default_variable$);\n"
        "  }\n"
        "  $field_member$.Set$lite$(\n"
        "      $default_variable$, std::RawString(\n"
        "        reinterpret_cast<const char*>(value), size),\n"
        "      GetArenaNoVirtual());\n"
        "  // @@protoc_insertion_point(field_set_pointer:$full_name$)\n"
        "}\n"
        "inline "
        "void $classname$::set_$name$(const std::shared_ptr<const char>& value,\n"
        "                             size_t size) {\n"
        "  if (!has_$name$()) {\n"
        "    clear_$oneof_name$();\n"
        "    set_has_$name$();\n"
        "    $field_member$.UnsafeSetDefault($default_variable$);\n"
        "  }\n"
        "  $field_member$.Set$lite$(\n"
        "      $default_variable$, std::RawString(value, size),\n"
        "      GetArenaNoVirtual());\n"
        "  // @@protoc_insertion_point(field_set_pointer:$full_name$)\n"
        "}\n"
        "inline void $classname$::set_$name$(const std::RawString& value) {\n"
        "  if (!has_$name$()) {\n"
        "    clear_$oneof_name$();\n"
        "    set_has_$name$();\n"
        "    $field_member$.UnsafeSetDefault($default_variable$);\n"
        "  }\n"
        "  $field_member$.Set$lite$($default_variable$,\n"
        "      value, GetArenaNoVirtual());\n"
        "  // @@protoc_insertion_point(field_set_char:$full_name$)\n"
        "}\n"
    );
    
    format(
        "inline std::RawString* $classname$::mutable_$name$() {\n"
        "  if (!has_$name$()) {\n"
        "    clear_$oneof_name$();\n"
        "    set_has_$name$();\n"
        "    $field_member$.UnsafeSetDefault($default_variable$);\n"
        "  }\n"
        "  return $field_member$.Mutable($default_variable$,\n"
        "      GetArenaNoVirtual());\n"
        "  // @@protoc_insertion_point(field_mutable:$full_name$)\n"
        "}\n"
        "inline std::RawString* $classname$::$release_name$() {\n"
        "  // @@protoc_insertion_point(field_release:$full_name$)\n"
        "  if (has_$name$()) {\n"
        "    clear_has_$oneof_name$();\n"
        "    return $field_member$.Release($default_variable$,\n"
        "        GetArenaNoVirtual());\n"
        "  } else {\n"
        "    return nullptr;\n"
        "  }\n"
        "}\n"
        "inline void $classname$::set_allocated_$name$(std::RawString* $name$) {\n"
        "  if (has_$oneof_name$()) {\n"
        "    clear_$oneof_name$();\n"
        "  }\n"
        "  if ($name$ != nullptr) {\n"
        "    set_has_$name$();\n"
        "    $field_member$.UnsafeSetDefault($name$);\n"
        "  }\n"
        "  // @@protoc_insertion_point(field_set_allocated:$full_name$)\n"
        "}\n");
    
    format(
          "inline std::RawString* $classname$::unsafe_arena_release_$name$() {\n"
          "  // "
          "@@protoc_insertion_point(field_unsafe_arena_release:$full_name$)\n"
          "  $DCHK$(GetArenaNoVirtual() != nullptr);\n"
          "  if (has_$name$()) {\n"
          "    clear_has_$oneof_name$();\n"
          "    return $field_member$.UnsafeArenaRelease(\n"
          "        $default_variable$, GetArenaNoVirtual());\n"
          "  } else {\n"
          "    return nullptr;\n"
          "  }\n"
          "}\n"
          "inline void $classname$::unsafe_arena_set_allocated_$name$("
          "std::RawString* $name$) {\n"
          "  $DCHK$(GetArenaNoVirtual() != nullptr);\n"
          "  if (!has_$name$()) {\n"
          "    $field_member$.UnsafeSetDefault($default_variable$);\n"
          "  }\n"
          "  clear_$oneof_name$();\n"
          "  if ($name$) {\n"
          "    set_has_$name$();\n"
          "    $field_member$.UnsafeArenaSetAllocated($default_variable$, "
          "$name$, GetArenaNoVirtual());\n"
          "  }\n"
          "  // @@protoc_insertion_point(field_unsafe_arena_set_allocated:"
          "$full_name$)\n"
          "}\n");
  } else {
    // No-arena case.
    format(
        "inline const std::RawString& $classname$::$name$() const {\n"
        "  // @@protoc_insertion_point(field_get:$full_name$)\n"
        "  if (has_$name$()) {\n"
        "    return $field_member$.GetNoArena();\n"
        "  }\n"
        "  return *$default_variable$;\n"
        "}\n"
    );

    format(
       "inline void $classname$::set_$name$(const std::string& value) {\n"
        "  // @@protoc_insertion_point(field_set:$full_name$)\n"
        "  if (!has_$name$()) {\n"
        "    clear_$oneof_name$();\n"
        "    set_has_$name$();\n"
        "    $field_member$.UnsafeSetDefault($default_variable$);\n"
        "  }\n"
        "  $field_member$.SetNoArena($default_variable$, std::RawString(value));\n"
        "  // @@protoc_insertion_point(field_set:$full_name$)\n"
        "}\n"
        "inline void $classname$::set_$name$(std::string&& value) {\n"
        "  // @@protoc_insertion_point(field_set:$full_name$)\n"
        "  if (!has_$name$()) {\n"
        "    clear_$oneof_name$();\n"
        "    set_has_$name$();\n"
        "    $field_member$.UnsafeSetDefault($default_variable$);\n"
        "  }\n"
        "  $field_member$.SetNoArena($default_variable$, std::RawString(::std::move(value)));\n"
        "  // @@protoc_insertion_point(field_set_rvalue:$full_name$)\n"
        "}\n"
        "inline void $classname$::set_$name$(const char* value) {\n"
        "  $null_check$"
        "  if (!has_$name$()) {\n"
        "    clear_$oneof_name$();\n"
        "    set_has_$name$();\n"
        "    $field_member$.UnsafeSetDefault($default_variable$);\n"
        "  }\n"
        "  $field_member$.SetNoArena($default_variable$,\n"
        "      std::RawString(value));\n"
        "  // @@protoc_insertion_point(field_set_char:$full_name$)\n"
        "}\n"
        "inline "
        "void $classname$::set_$name$(const $pointer_type$* value, size_t "
        "size) {\n"
        "  if (!has_$name$()) {\n"
        "    clear_$oneof_name$();\n"
        "    set_has_$name$();\n"
        "    $field_member$.UnsafeSetDefault($default_variable$);\n"
        "  }\n"
        "  $field_member$.SetNoArena($default_variable$, std::RawString(\n"
        "      reinterpret_cast<const char*>(value), size));\n"
        "  // @@protoc_insertion_point(field_set_pointer:$full_name$)\n"
        "}\n"
        "inline "
        "void $classname$::set_$name$(const std::shared_ptr<const char>& value, size_t "
        "size) {\n"
        "  if (!has_$name$()) {\n"
        "    clear_$oneof_name$();\n"
        "    set_has_$name$();\n"
        "    $field_member$.UnsafeSetDefault($default_variable$);\n"
        "  }\n"
        "  $field_member$.SetNoArena($default_variable$, std::RawString(value, size));\n"
        "  // @@protoc_insertion_point(field_set_pointer:$full_name$)\n"
        "}\n"
        "inline void $classname$::set_$name$(const std::RawString& value) {\n"
        "  if (!has_$name$()) {\n"
        "    clear_$oneof_name$();\n"
        "    set_has_$name$();\n"
        "    $field_member$.UnsafeSetDefault($default_variable$);\n"
        "  }\n"
        "  $field_member$.SetNoArena($default_variable$, value);\n"
        "  // @@protoc_insertion_point(field_set_char:$full_name$)\n"
        "}\n"
    );
      
    format(
        "inline std::RawString* $classname$::mutable_$name$() {\n"
        "  if (!has_$name$()) {\n"
        "    clear_$oneof_name$();\n"
        "    set_has_$name$();\n"
        "    $field_member$.UnsafeSetDefault($default_variable$);\n"
        "  }\n"
        "  // @@protoc_insertion_point(field_mutable:$full_name$)\n"
        "  return $field_member$.MutableNoArena($default_variable$);\n"
        "}\n"
        "inline std::RawString* $classname$::$release_name$() {\n"
        "  // @@protoc_insertion_point(field_release:$full_name$)\n"
        "  if (has_$name$()) {\n"
        "    clear_has_$oneof_name$();\n"
        "    return $field_member$.ReleaseNoArena($default_variable$);\n"
        "  } else {\n"
        "    return nullptr;\n"
        "  }\n"
        "}\n"
        "inline void $classname$::set_allocated_$name$(std::RawString* $name$) {\n"
        "  if (has_$oneof_name$()) {\n"
        "    clear_$oneof_name$();\n"
        "  }\n"
        "  if ($name$ != nullptr) {\n"
        "    set_has_$name$();\n"
        "    $field_member$.UnsafeSetDefault($name$);\n"
        "  }\n"
        "  // @@protoc_insertion_point(field_set_allocated:$full_name$)\n"
        "}\n");
  }
}

void RawStringOneofFieldGenerator::GenerateClearingCode(
    io::Printer* printer) const {
  Formatter format(printer, variables_);
  if (SupportsArenas(descriptor_)) {
    format(
        "$field_member$.Destroy($default_variable$,\n"
        "    GetArenaNoVirtual());\n");
  } else {
    format("$field_member$.DestroyNoArena($default_variable$);\n");
  }
}

void RawStringOneofFieldGenerator::GenerateMessageClearingCode(
    io::Printer* printer) const {
  return GenerateClearingCode(printer);
}

void RawStringOneofFieldGenerator::GenerateSwappingCode(
    io::Printer* printer) const {
  // Don't print any swapping code. Swapping the union will swap this field.
}

void RawStringOneofFieldGenerator::GenerateConstructorCode(
    io::Printer* printer) const {
  Formatter format(printer, variables_);
  format(
      "$ns$::_$classname$_default_instance_.$name$_.UnsafeSetDefault(\n"
      "    $default_variable$);\n");
}

void RawStringOneofFieldGenerator::GenerateDestructorCode(
    io::Printer* printer) const {
  Formatter format(printer, variables_);
  format(
      "if (has_$name$()) {\n"
      "  $field_member$.DestroyNoArena($default_variable$);\n"
      "}\n");
}

void RawStringOneofFieldGenerator::GenerateMergeFromCodedStream(
    io::Printer* printer) const {
  Formatter format(printer, variables_);
  format(
        "DO_(::$proto_ns$::internal::WireFormatLite::Read$declared_type$(\n"
        "      input, this->mutable_$name$()));\n");
}

// ===================================================================

RepeatedRawStringFieldGenerator::RepeatedRawStringFieldGenerator(
    const FieldDescriptor* descriptor, const Options& options)
    : FieldGenerator(descriptor, options) {
  SetRawStringVariables(descriptor, &variables_, options);
}

RepeatedRawStringFieldGenerator::~RepeatedRawStringFieldGenerator() {}

void RepeatedRawStringFieldGenerator::GeneratePrivateMembers(
    io::Printer* printer) const {
  Formatter format(printer, variables_);
  format("::$proto_ns$::RepeatedPtrField<std::RawString> $name$_;\n");
}

void RepeatedRawStringFieldGenerator::GenerateAccessorDeclarations(
    io::Printer* printer) const {
  Formatter format(printer, variables_);

  format(
      "$deprecated_attr$const std::RawString& ${1$$name$$}$(int index) const;\n",
      descriptor_);
  
  format(
      "$deprecated_attr$std::RawString* ${1$mutable_$name$$}$(int index);\n"
      "$deprecated_attr$void ${1$set_$name$$}$(int index, const "
      "std::string& value);\n"
      "$deprecated_attr$void ${1$set_$name$$}$(int index, std::string&& "
      "value);\n"
      "$deprecated_attr$void ${1$set_$name$$}$(int index, const "
      "char* value);\n"
      "$deprecated_attr$void ${1$set_$name$$}$("
      "int index, const $pointer_type$* value, size_t size);\n"
      "$deprecated_attr$void ${1$set_$name$$}$("
      "int index, const std::shared_ptr<const char>& value, size_t size);\n"
      "$deprecated_attr$void ${1$set_$name$$}$(int index, const "
      "std::RawString& value);\n",
      descriptor_);
 
  format(
      "$deprecated_attr$std::RawString* ${1$add_$name$$}$();\n"
      "$deprecated_attr$void ${1$add_$name$$}$(const std::string& value);\n"
      "$deprecated_attr$void ${1$add_$name$$}$(std::string&& value);\n"
      "$deprecated_attr$void ${1$add_$name$$}$(const char* value);\n"
      "$deprecated_attr$void ${1$add_$name$$}$(const $pointer_type$* "
      "value, size_t size)"
      ";\n"
      "$deprecated_attr$void ${1$add_$name$$}$(const std::shared_ptr<const char>& "
      "value, size_t size)"
      ";\n"
      "$deprecated_attr$void ${1$add_$name$$}$(const std::RawString& value);\n",
      descriptor_);
  
  format(
      "$deprecated_attr$const ::$proto_ns$::RepeatedPtrField<std::RawString>& "
      "${1$$name$$}$() "
      "const;\n"
      "$deprecated_attr$::$proto_ns$::RepeatedPtrField<std::RawString>* "
      "${1$mutable_$name$$}$()"
      ";\n",
      descriptor_);
}

void RepeatedRawStringFieldGenerator::GenerateInlineAccessorDefinitions(
    io::Printer* printer) const {
  Formatter format(printer, variables_);
  if (options_.safe_boundary_check) {
    format(
        "inline const std::RawString& $classname$::$name$(int index) const {\n"
        "  // @@protoc_insertion_point(field_get:$full_name$)\n"
        "  return $name$_.InternalCheckedGet(\n"
        "      index, ::$proto_ns$::internal::GetEmptyStringAlreadyInited());\n"
        "}\n");
  } else {
    format(
        "inline const std::RawString& $classname$::$name$(int index) const {\n"
        "  // @@protoc_insertion_point(field_get:$full_name$)\n"
        "  return $name$_.Get(index);\n"
        "}\n");
  }

  format(
      "inline std::RawString* $classname$::mutable_$name$(int index) {\n"
      "  // @@protoc_insertion_point(field_mutable:$full_name$)\n"
      "  return $name$_.Mutable(index);\n"
      "}\n"
      "inline void $classname$::set_$name$(int index, const std::string& "
      "value) "
      "{\n"
      "  // @@protoc_insertion_point(field_set:$full_name$)\n"
      "  $name$_.Mutable(index)->assign(value);\n"
      "}\n"
      "inline void $classname$::set_$name$(int index, std::string&& value) {\n"
      "  // @@protoc_insertion_point(field_set:$full_name$)\n"
      "  $name$_.Mutable(index)->assign(std::move(value));\n"
      "}\n"
      "inline void $classname$::set_$name$(int index, const char* value) {\n"
      "  $null_check$"
      "  $name$_.Mutable(index)->assign(value);\n"
      "  // @@protoc_insertion_point(field_set_char:$full_name$)\n"
      "}\n"
      "inline void "
      "$classname$::set_$name$"
      "(int index, const $pointer_type$* value, size_t size) {\n"
      "  $name$_.Mutable(index)->assign(\n"
      "    reinterpret_cast<const char*>(value), size);\n"
      "  // @@protoc_insertion_point(field_set_pointer:$full_name$)\n"
      "}\n"
      "inline void "
      "$classname$::set_$name$"
      "(int index, const std::shared_ptr<const char>& value, size_t size) {\n"
      "  $name$_.Mutable(index)->assign(value, size);\n"
      "  // @@protoc_insertion_point(field_set_pointer:$full_name$)\n"
      "}\n"
      "inline void $classname$::set_$name$(int index, const std::RawString& value) {\n"
      "  $name$_.Mutable(index)->assign(value);\n"
      "  // @@protoc_insertion_point(field_set_char:$full_name$)\n"
      "}\n");
  
  format(
      "inline std::RawString* $classname$::add_$name$() {\n"
      "  // @@protoc_insertion_point(field_add_mutable:$full_name$)\n"
      "  return $name$_.Add();\n"
      "}\n"
      "inline void $classname$::add_$name$(const std::string& value) {\n"
      "  $name$_.Add()->assign(value);\n"
      "  // @@protoc_insertion_point(field_add:$full_name$)\n"
      "}\n"
      "inline void $classname$::add_$name$(std::string&& value) {\n"
      "  $name$_.Add()->assign(std::move(value));\n"
      "  // @@protoc_insertion_point(field_add:$full_name$)\n"
      "}\n"
      "inline void $classname$::add_$name$(const char* value) {\n"
      "  $null_check$"
      "  $name$_.Add()->assign(value);\n"
      "  // @@protoc_insertion_point(field_add_char:$full_name$)\n"
      "}\n"
      "inline void "
      "$classname$::add_$name$(const $pointer_type$* value, size_t size) {\n"
      "  $name$_.Add()->assign(reinterpret_cast<const char*>(value), size);\n"
      "  // @@protoc_insertion_point(field_add_pointer:$full_name$)\n"
      "}\n"
      "inline void "
      "$classname$::add_$name$(const std::shared_ptr<const char>& value, size_t size) {\n"
      "  $name$_.Add()->assign(value, size);\n"
      "  // @@protoc_insertion_point(field_add_pointer:$full_name$)\n"
      "}\n"
      "inline void $classname$::add_$name$(const std::RawString& value) {\n"
      "  $name$_.Add()->assign(value);\n"
      "  // @@protoc_insertion_point(field_add_char:$full_name$)\n"
      "}\n");
  
  format(
      "inline const ::$proto_ns$::RepeatedPtrField<std::RawString>&\n"
      "$classname$::$name$() const {\n"
      "  // @@protoc_insertion_point(field_list:$full_name$)\n"
      "  return $name$_;\n"
      "}\n"
      "inline ::$proto_ns$::RepeatedPtrField<std::RawString>*\n"
      "$classname$::mutable_$name$() {\n"
      "  // @@protoc_insertion_point(field_mutable_list:$full_name$)\n"
      "  return &$name$_;\n"
      "}\n");
}

void RepeatedRawStringFieldGenerator::GenerateClearingCode(
    io::Printer* printer) const {
  Formatter format(printer, variables_);
  format("$name$_.Clear();\n");
}

void RepeatedRawStringFieldGenerator::GenerateMergingCode(
    io::Printer* printer) const {
  Formatter format(printer, variables_);
  format("$name$_.MergeFrom(from.$name$_);\n");
}

void RepeatedRawStringFieldGenerator::GenerateSwappingCode(
    io::Printer* printer) const {
  Formatter format(printer, variables_);
  format("$name$_.InternalSwap(CastToBase(&other->$name$_));\n");
}

void RepeatedRawStringFieldGenerator::GenerateConstructorCode(
    io::Printer* printer) const {
  // Not needed for repeated fields.
}

void RepeatedRawStringFieldGenerator::GenerateCopyConstructorCode(
    io::Printer* printer) const {
  Formatter format(printer, variables_);
  format("$name$_.CopyFrom(from.$name$_);");
}

void RepeatedRawStringFieldGenerator::GenerateMergeFromCodedStream(
    io::Printer* printer) const {
  Formatter format(printer, variables_);
  format(
      "DO_(::$proto_ns$::internal::WireFormatLite::Read$declared_type$(\n"
      "      input, this->add_$name$()));\n");
  if (descriptor_->type() == FieldDescriptor::TYPE_STRING) {
    GenerateUtf8CheckCodeForString(
        descriptor_, options_, true,
        "this->$name$(this->$name$_size() - 1).data(),\n"
        "static_cast<int>(this->$name$(this->$name$_size() - 1).size()),\n",
        format);
  }
}

void RepeatedRawStringFieldGenerator::GenerateSerializeWithCachedSizes(
    io::Printer* printer) const {
  Formatter format(printer, variables_);
  format("for (int i = 0, n = this->$name$_size(); i < n; i++) {\n");
  format.Indent();
  if (descriptor_->type() == FieldDescriptor::TYPE_STRING) {
    GenerateUtf8CheckCodeForString(
        descriptor_, options_, false,
        "this->$name$(i).data(), static_cast<int>(this->$name$(i).size()),\n",
        format);
  }
  format.Outdent();
  format(
      "  ::$proto_ns$::internal::WireFormatLite::Write$declared_type$(\n"
      "    $number$, this->$name$(i), output);\n"
      "}\n");
}

void RepeatedRawStringFieldGenerator::GenerateSerializeWithCachedSizesToArray(
    io::Printer* printer) const {
  Formatter format(printer, variables_);
  format("for (int i = 0, n = this->$name$_size(); i < n; i++) {\n");
  format.Indent();
  if (descriptor_->type() == FieldDescriptor::TYPE_STRING) {
    GenerateUtf8CheckCodeForString(
        descriptor_, options_, false,
        "this->$name$(i).data(), static_cast<int>(this->$name$(i).size()),\n",
        format);
  }
  format.Outdent();
  format(
      "  target = ::$proto_ns$::internal::WireFormatLite::\n"
      "    Write$declared_type$ToArray($number$, this->$name$(i), target);\n"
      "}\n");
}

void RepeatedRawStringFieldGenerator::GenerateByteSize(
    io::Printer* printer) const {
  Formatter format(printer, variables_);
  format(
      "total_size += $tag_size$ *\n"
      "    ::$proto_ns$::internal::FromIntSize(this->$name$_size());\n"
      "for (int i = 0, n = this->$name$_size(); i < n; i++) {\n"
      "  total_size += "
      "::$proto_ns$::internal::WireFormatLite::$declared_type$Size(\n"
      "    this->$name$(i));\n"
      "}\n");
}

}  // namespace cpp
}  // namespace compiler
}  // namespace protobuf
}  // namespace google
