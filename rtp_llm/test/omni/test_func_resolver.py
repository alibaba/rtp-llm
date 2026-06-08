import unittest


class TestResolveFunc(unittest.TestCase):
    def test_resolve_valid_function(self):
        from rtp_llm.omni.engine.func_resolver import resolve_func

        func = resolve_func("os.path.join")
        self.assertTrue(callable(func))
        self.assertEqual(func("a", "b"), "a/b")

    def test_resolve_invalid_module(self):
        from rtp_llm.omni.engine.func_resolver import resolve_func

        with self.assertRaises(ModuleNotFoundError):
            resolve_func("nonexistent.module.func")

    def test_resolve_invalid_attribute(self):
        from rtp_llm.omni.engine.func_resolver import resolve_func

        with self.assertRaises(AttributeError):
            resolve_func("os.path.nonexistent_func")

    def test_resolve_no_dot(self):
        from rtp_llm.omni.engine.func_resolver import resolve_func

        with self.assertRaises(ValueError):
            resolve_func("just_a_name")

    def test_resolve_not_callable(self):
        from rtp_llm.omni.engine.func_resolver import resolve_func

        with self.assertRaises(TypeError):
            resolve_func("os.path.sep")


if __name__ == "__main__":
    unittest.main()
