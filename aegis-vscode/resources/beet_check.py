import importlib.util;

spec = importlib.util.find_spec("beet")

if spec is not None:
    import beet
    print(True, beet.__version__)
else:
    print(False)