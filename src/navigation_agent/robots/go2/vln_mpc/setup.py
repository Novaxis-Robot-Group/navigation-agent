from setuptools import find_packages, setup

package_name = "vln_mpc"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=("test",)),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools", "casadi>=3.7,<4"],
    zip_safe=True,
    maintainer="xiaoyang",
    maintainer_email="xiaoyang@lightrobo.com",
    description="Capture-time-aligned MPC controller for VLN responses.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={"console_scripts": ["vln_mpc = vln_mpc.mpc_node:main"]},
)
