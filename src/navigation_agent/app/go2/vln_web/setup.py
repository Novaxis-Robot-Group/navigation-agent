from glob import glob

from setuptools import find_packages, setup

package_name = "vln_web"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/web", glob("web/*")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="xiaoyang",
    maintainer_email="xiaoyang@lightrobo.com",
    description="Browser interface for testing and monitoring the VLN pipeline.",
    license="Apache-2.0",
    entry_points={"console_scripts": ["vln_web = vln_web.web_node:main"]},
)
