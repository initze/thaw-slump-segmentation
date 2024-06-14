# taken from https://github.com/mkdocstrings/mkdocstrings/issues/179#issuecomment-784631904
from pathlib import Path
import mkdocs_gen_files

src_root = Path("src/")
for path in src_root.glob("**/*.py"):
    if path.stem == '__init__':
      continue
    if path.parent.stem == 'unet3p':
      continue

    rel_path = path.relative_to(Path('.'))
    doc_path = Path("reference", rel_path).with_suffix(".md")
    with mkdocs_gen_files.open(doc_path, "w") as f:
      ident = ".".join(rel_path.with_suffix("").parts)
      print("::: " + ident, file=f)
      print("::: " + ident)
