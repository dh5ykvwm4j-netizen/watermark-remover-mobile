# 视频去水印 — 手机版

## 📱 方案一：浏览器直接使用（无需安装）

打开 `index.html` 即可在手机浏览器中使用。

**注意**：浏览器版受限于 Web 环境，完整视频导出功能建议使用 APK 版本。
OpenCV.js 需要从 CDN 加载，首次使用需要联网。

## 📦 方案二：APK 安装包（推荐）

### 方法 A：用 GitHub Actions 免费构建（推荐）

1. 将本文件夹上传到你的 GitHub 仓库
2. 进入仓库 Actions 页面，手动运行 "Build Android APK" 工作流
3. 约 30-60 分钟后，下载生成的 APK 文件

### 方法 B：本地 WSL + Buildozer 构建

```bash
# 1. 安装 WSL2 Ubuntu
wsl --install -d Ubuntu-24.04

# 2. 在 Ubuntu 中执行
sudo apt update
sudo apt install -y git zip unzip openjdk-17-jdk python3-pip autoconf libtool pkg-config zlib1g-dev libncurses5-dev libncursesw5-dev libtinfo5 cmake libffi-dev libssl-dev
pip3 install --upgrade pip buildozer cython

# 3. 复制本文件夹到 Ubuntu 中，进入目录执行
buildozer android debug

# 4. APK 生成在 bin/ 目录下
```

### 方法 C：使用 Google Colab

在 Google Colab 中运行以下命令：

```python
!pip install buildozer cython
!git clone https://github.com/你的仓库/watermark-remover-mobile
%cd watermark-remover-mobile
!buildozer android debug
```

## ⚙️ 功能说明

- **加载视频**：从手机存储选择视频文件
- **框选水印**：手指在画面拖动绘制矩形框
- **取色**：自动提取框中心颜色作为水印样本
- **容差/半径**：调节颜色匹配灵敏度与修复范围
- **预览**：查看当前帧去水印效果
- **处理**：对整段视频逐帧去除水印
