from setuptools import setup, find_packages

setup(
    name="odbargo_app",
    version="0.0.6",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "fastapi[standard]<=0.115.12",
        "uvicorn[standard]<=0.34.1",
        "argopy==1.1.0",
        "pydantic<=2.11.3",
        "numpy<2.0.0,>=1.26.4"
    ],
    entry_points={
        "console_scripts": [
            "odbargo_app=argo_app.app:main",
        ],
    },
)
