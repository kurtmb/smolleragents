from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="smolleragents",
    version="1.0.0",
    author="Kurt Boden",
    author_email="smolleragents@gmail.com",
    description="A lightweight ReACT agent framework with OpenAI API integration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kurtmb/smolleragents",
    project_urls={
        "Bug Reports": "https://github.com/kurtmb/smolleragents/issues",
        "Source": "https://github.com/kurtmb/smolleragents",
        "Documentation": "https://github.com/kurtmb/smolleragents#readme",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    install_requires=[
        "openai>=1.0.0",
        "pydantic>=2.0.0",
        "requests>=2.25.0",
        "duckduckgo-search>=4.0.0",
        "beautifulsoup4>=4.9.0",
        "lxml>=4.6.0",
        "jinja2>=3.0.0",
        "pyyaml>=6.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
        ]
    },
) 