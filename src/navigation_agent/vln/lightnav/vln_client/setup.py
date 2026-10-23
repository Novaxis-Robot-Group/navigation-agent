from setuptools import find_packages, setup

package_name = "vln_client"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=("test",)),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="xiaoyang",
    maintainer_email="xiaoyang@lightrobo.com",
    description="ROS 2 node for the VLN client.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={"console_scripts": ["vln_client = vln_client.vln_node:main"]},
)
