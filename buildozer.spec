[app]
title = WatermarkRemover
package.name = waterremover
package.domain = com.xly
source.dir = .
source.include_exts = py,png,jpg,jpeg,gif,ttf,mp4
version = 1.0.0
requirements = python3,kivy==2.3.0,pillow,numpy==1.26.4,pyjnius,certifi
orientation = landscape
osx.python_version = 3
osx.kivy_version = 2.2.1
fullscreen = 0

# Android
android.api = 34
android.minapi = 24
android.ndk = 26
android.archs = arm64-v8a
android.accept_sdk_license = True
android.private_storage = True
android.permissions = READ_EXTERNAL_STORAGE,WRITE_EXTERNAL_STORAGE,READ_MEDIA_VIDEO
android.wakelock = True
android.enable_p4a = True
android.gradle_dependencies = 'androidx.core:core:1.12.0'

# iOS (skip)
ios.codesign.allowed = false

[buildozer]
log_level = 2
warn_on_root = 1
