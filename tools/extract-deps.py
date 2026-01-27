# extract testing dependencies from a pyproject.toml
import tomli
import shlex

data = tomli.load(open("pyproject.toml", "rb"))
deps = data["project"]["optional-dependencies"]["tests"]
print(" ".join(shlex.quote(dep) for dep in deps))
