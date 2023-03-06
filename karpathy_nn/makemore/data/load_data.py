import importlib.resources as pkg_resources

def load_names() -> list[str]:
    """Loads the names from the ``names.txt`` file as a list of strings."""
    # print(os.path.abspath("."))
    # with open("names.txt") as f:
    #     return f.readlines()
    data = pkg_resources.read_text("karpathy_nn.makemore.data", "names.txt")
    data = data.splitlines()

    return data

def load_shakespeare() -> str:
    """Returns a concatenation of all works of Shakespeare as a single string."""
    return pkg_resources.read_text("karpathy_nn.makemore.data", "shakespeare.txt")