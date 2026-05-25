import setuptools

# 读取README.md作为长描述
with open("README.md", "r") as f:
    long_description = f.read()

# 读取依赖列表requirements.txt
# 忽略#开头或者版本号不明确指定的条目
with open("requirements.txt", "r") as f:
    requirements = [req.strip() for req in f.readlines() if not req.startswith("#") and req.__contains__("==")]

setuptools.setup(
    name="galfitx",
    version="0.0.1",  # 版本号
    author="Chao Ma",  # 作者
    author_email="machao@pku.edu.cn",  # 邮箱
    description="galfitx package.",  # 短描述
    long_description=long_description,  # 长描述
    long_description_content_type="text/markdown",  # 长描述类型
    url="https://csst-tb.bao.ac.cn/code/csst-l1/csst_common",  # 主页
    packages=setuptools.find_packages(where="."),  # 用setuptools工具自动发现带有__init__.py的包
    license="MIT",  # 证书类型
    classifiers=[  # 程序分类, 参考 https://pypi.org/classifiers/
        # How mature is this project?
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Astronomy",
    ],
    include_package_data=True,  # 设置包含随包数据
    package_data={  # 具体随包数据路径
        "galfitx": ["data/*"],
    },
    install_requires=requirements,
    python_requires=">=3.11",
)
