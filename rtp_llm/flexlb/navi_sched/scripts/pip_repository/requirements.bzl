"""Analysis-only Python labels needed by NaviSched package targets."""

def requirement(name):
    if name != "thrift":
        fail("unexpected Python wheel dependency in NaviSched package build: %s" % name)
    return Label("@pip//:thrift")
