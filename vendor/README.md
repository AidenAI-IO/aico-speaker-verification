# Vendored ERes2Net implementation

`3d-speaker/speakerlab/models/eres2net/{ERes2Net.py,fusion.py,pooling_layers.py}`
are copied verbatim from the upstream [3D-Speaker repository](https://github.com/modelscope/3D-Speaker)
at revision `065629c313eaf1a01c65c640c46d77e61e9607b4` (Apache-2.0).
Only minimal `__init__.py` files were added here to make the subpackage importable.
The upstream license is in `../licenses/3D-Speaker-LICENSE`.

`export_eres2net.py` imports this pinned implementation explicitly instead of
relying on a changing global `speakerlab` package. Modifications to the model
source should be clearly identified here and in the changed files.
