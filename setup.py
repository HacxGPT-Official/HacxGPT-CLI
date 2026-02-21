from setuptools import setup, find_packages

setup(
    name="hacxgpt",
    version="2.1.0",
    description="Advanced Uncensored AI Terminal Interface",
    author="BlackTechX",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "hacxgpt": ["providers.json", "prompts/*.md"],
    },
    install_requires=[
        "rich",
        "python-dotenv",
        "pwinput",
        "pyperclip",
        "colorama",
        "prompt_toolkit",
        "requests"
    ],
    entry_points={
        "console_scripts": [
            "hacxgpt=hacxgpt.main:main",
        ],
    },
    python_requires=">=3.8",
)

