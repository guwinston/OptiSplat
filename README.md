# OptiSplat


## 环境依赖：
### linux:
1. x11
    ```bash
    sudo apt-get update
    sudo apt-get install libx11-dev libxcursor-dev libxinerama-dev libxrandr-dev libxi-dev
    ```
2. OpenGL
    ```bash
    sudo apt-get update
    sudo apt-get install libgl1-mesa-dev
    ```

## 用法
1. 使用vscode cmake插件
2. 使用命令行
    ```bash
    mkdir build
    cd build
    cmake ..
    cmake --build . --config RelWithDebInfo
    ./optisplat_test
    ```