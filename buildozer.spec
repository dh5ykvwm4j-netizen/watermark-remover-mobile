[app]
title = 视频去水印
package.name = waterremover
package.domain = com.xly
source.dir = .
source.include_exts = py,png,jpg,jpeg,gif,ttf,mp4
version = 1.0.0
requirements = python3,kivy==2.3.0,pillow,numpy,pyjnius,certifi
orientation = landscape
osx.python_version = 3
osx.kivy_version = 2.2.1
fullscreen = 0

# Android
android.api = 34
android.minapi = 24
android.sdk = 34
android.ndk = 27
android.accept_sdk_license = True
android.private_storage = True
android.permissions = READ_EXTERNAL_STORAGE,WRITE_EXTERNAL_STORAGE,READ_MEDIA_VIDEO
android.wakelock = True
android.enable_p4a = True
android.gradle_dependencies = 'androidx.core:core:1.12.0'
android.add_src = 
android.add_src_layout = 
android.add_src_mime_types = 

# iOS (skip)
ios.codesign.allowed = false

[buildozer]
log_level = 2
warn_on_root = 1
