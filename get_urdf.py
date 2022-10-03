from object2urdf import ObjectUrdfBuilder

object_folder = "cubes"
builder = ObjectUrdfBuilder(object_folder, urdf_prototype="_prototype.urdf")
builder.build_urdf(
    filename="cubes/cube7.stl",
    force_overwrite=True,
    decompose_concave=True,
    force_decompose=False,
    center="mass"
)