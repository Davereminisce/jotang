# Vmware.Linux虚拟机安装&vscode的远程连接&git安装

2024/10/13

## Vmware.Linux虚拟机

### Vmware虚拟机安装

网盘下载链接https://pan.quark.cn/s/b3e13befc19f#/list/share

![{B91F667B-5195-4C8F-8DE8-A0F8BA1ACBF6}.png](https://github.com/Davereminisce/image/blob/master/%7BB91F667B-5195-4C8F-8DE8-A0F8BA1ACBF6%7D.png?raw=true)

选择17.6.0版本

![{44968394-B9C2-4122-B759-8B4E76EEFD45}.png](https://github.com/Davereminisce/image/blob/master/%7B44968394-B9C2-4122-B759-8B4E76EEFD45%7D.png?raw=true)

点击exe文件进行下载

![{BDA94670-4976-4F9A-8F04-F05A090ED63B}.png](https://github.com/Davereminisce/image/blob/master/%7BBDA94670-4976-4F9A-8F04-F05A090ED63B%7D.png?raw=true)

按照安装步骤进行下载即可

![{DA6CAB4C-E7C1-4B20-99D5-8A7417252204}.png](https://github.com/Davereminisce/image/blob/master/%7BDA6CAB4C-E7C1-4B20-99D5-8A7417252204%7D.png?raw=true)

建议更改安装位置![{98AFD468-FFBD-47DC-914D-7B37AC6EE73D}.png](https://github.com/Davereminisce/image/blob/master/%7B98AFD468-FFBD-47DC-914D-7B37AC6EE73D%7D.png?raw=true)

取消勾选

![{08297C65-AAAE-463B-A9BD-DEE7E2181AFB}.png](https://github.com/Davereminisce/image/blob/master/%7B08297C65-AAAE-463B-A9BD-DEE7E2181AFB%7D.png?raw=true)

点击许可证并输入密钥

至此安装部分完成

### 配置虚拟机

首先需要更改虚拟机存放位置

![{35B6C2B5-CA28-4DF1-BC28-8DC1AC016F19}.png](https://github.com/Davereminisce/image/blob/master/%7B35B6C2B5-CA28-4DF1-BC28-8DC1AC016F19%7D.png?raw=true)

点击编辑选择首选项

![{73CF6012-CEAE-4D35-B9DC-9F7514B042F2}.png](https://github.com/Davereminisce/image/blob/master/%7B73CF6012-CEAE-4D35-B9DC-9F7514B042F2%7D.png?raw=true)

点击确定，虚拟机都会安装到此路径，方便进行管理。

### Linux虚拟机的安装

Ubuntu下载地址https://mirrors.tuna.tsinghua.edu.cn/ubuntu-releases/22.04.4/

![{7DCFFFE1-9107-4B83-9C42-958D6094F232}.png](https://github.com/Davereminisce/image/blob/master/%7B7DCFFFE1-9107-4B83-9C42-958D6094F232%7D.png?raw=true)

选择安装ubuntu-22.04.5-desktop-amd64.iso

![{1ED13277-03C7-4A4E-B24E-DB4BEADF0ED7}.png](https://github.com/Davereminisce/image/blob/master/%7B1ED13277-03C7-4A4E-B24E-DB4BEADF0ED7%7D.png?raw=true)

点击创建新的虚拟机

![{CA34CCD7-3FCA-40DC-A492-6A206FB321AC}.png](https://github.com/Davereminisce/image/blob/master/%7BCA34CCD7-3FCA-40DC-A492-6A206FB321AC%7D.png?raw=true)

直接点击下一步

![{D8A04EDD-452C-44E5-8823-1FCAEE7CB73E}.png](https://github.com/Davereminisce/image/blob/master/%7BD8A04EDD-452C-44E5-8823-1FCAEE7CB73E%7D.png?raw=true)

选择我们下载的ubuntu地址

![{BA3BA684-C6D5-4D14-A04D-14CBF691C848}.png](https://github.com/Davereminisce/image/blob/master/%7BBA3BA684-C6D5-4D14-A04D-14CBF691C848%7D.png?raw=true)

自主设置即可

![{2C8A868F-279E-452A-9137-CDFD25D6A737}.png](https://github.com/Davereminisce/image/blob/master/%7B2C8A868F-279E-452A-9137-CDFD25D6A737%7D.png?raw=true)

更改目录至之前设置的data存放路径（还是为了方便管理）

![{568FBE97-EFC7-41B0-9B12-8F4C7DF2B1D9}.png](https://github.com/Davereminisce/image/blob/master/%7B568FBE97-EFC7-41B0-9B12-8F4C7DF2B1D9%7D.png?raw=true)

如图设置即可![{BAA5A74A-9709-4336-AE75-7C4632C97ED3}.png](https://github.com/Davereminisce/image/blob/master/%7BBAA5A74A-9709-4336-AE75-7C4632C97ED3%7D.png?raw=true)

虚拟机开机后设置如下![{61FAC2B0-DE33-48CD-81D8-B079E044105A}.png](https://github.com/Davereminisce/image/blob/master/%7B61FAC2B0-DE33-48CD-81D8-B079E044105A%7D.png?raw=true)

选择中文

![{CDA8FCD6-060B-49FA-86EF-15F6D7891F26}.png](https://github.com/Davereminisce/image/blob/master/%7BCDA8FCD6-060B-49FA-86EF-15F6D7891F26%7D.png?raw=true)

选择最小安装

![{98C54191-DA6D-4F39-82D5-2F624249A3DA}.png](https://github.com/Davereminisce/image/blob/master/%7B98C54191-DA6D-4F39-82D5-2F624249A3DA%7D.png?raw=true)

选择清除磁盘并安装Ubuntu

![{0207AF59-1FE5-4A9F-9B03-72C129A238DC}.png](https://github.com/Davereminisce/image/blob/master/%7B0207AF59-1FE5-4A9F-9B03-72C129A238DC%7D.png?raw=true)

选择上海

![{A3E0AA41-9FF4-4D10-A6BA-85BF5D04E77E}.png](https://github.com/Davereminisce/image/blob/master/%7BA3E0AA41-9FF4-4D10-A6BA-85BF5D04E77E%7D.png?raw=true)

按照之前注册的用户填写即可

![{244EBE5D-305E-41EE-A476-214E00CBFA6F}.png](https://github.com/Davereminisce/image/blob/master/%7B244EBE5D-305E-41EE-A476-214E00CBFA6F%7D.png?raw=true)

点击Restart Now即可

## VS Code通过remote-ssh远程连接虚拟机

### Linux安装openssh-server

首先更新镜像源

![{683EDABE-C96C-43DA-8A35-D6C774645639}.png](https://github.com/Davereminisce/image/blob/master/%7B683EDABE-C96C-43DA-8A35-D6C774645639%7D.png?raw=true)

点击software&Updates

![{10915932-91B7-41C1-AD5D-E9FEF0FF4CA6}.png](https://github.com/Davereminisce/image/blob/master/%7B10915932-91B7-41C1-AD5D-E9FEF0FF4CA6%7D.png?raw=true)

更改Download from为http://mirrors.aliyun.com/ubuntu

![{42026C2D-9DD3-4AF5-A11F-AE07EC6B8F46}.png](https://github.com/Davereminisce/image/blob/master/%7B42026C2D-9DD3-4AF5-A11F-AE07EC6B8F46%7D.png?raw=true)

点击左下角然后点击Terminal

![{2DB2D99E-2259-4640-BD11-60322298740F}.png](https://github.com/Davereminisce/image/blob/master/%7B2DB2D99E-2259-4640-BD11-60322298740F%7D.png?raw=true)

该界面即是终端

在终端中输入以下代码：

`sudo apt-get remove openssh-server      # 卸载
sudo apt-get install openssh-server     # 安装
sudo service ssh --full-restart     # 重启ssh 服务
sudo systemctl enable ssh       # 自动启动`

完成后在终端输入 ssh -v 若如图显示则证明ssh安装成功

![{747D3EF6-350D-4295-A316-9A0B33575F0D}.png](https://github.com/Davereminisce/image/blob/master/%7B747D3EF6-350D-4295-A316-9A0B33575F0D%7D.png?raw=true)

按照vi修改规则编辑配置文件https://blog.csdn.net/wangzhicheng987/article/details/120921484?ops_request_misc=%257B%2522request%255Fid%2522%253A%25229625A636-FB80-4CB3-A2D1-8F43E38285CC%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=9625A636-FB80-4CB3-A2D1-8F43E38285CC&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~blog~sobaiduend~default-1-120921484-null-null.nonecase&utm_term=Linux%E4%B8%ADvi%E6%96%87%E6%9C%AC%E7%BC%96%E8%BE%91%E5%99%A8&spm=1018.2226.3001.4450

![{9FA7C5A6-066D-40EE-A5ED-C04EEE0EDC71}.png](https://github.com/Davereminisce/image/blob/master/%7B9FA7C5A6-066D-40EE-A5ED-C04EEE0EDC71%7D.png?raw=true)

进行如下添加

`Port 22
PermitRootLogin yes
PasswordAuthentication yes
AllowUsers xxx # 这里的 "xxx" 改成你自己的登陆用户名
RSAAuthentication yes
PubKeyAUthentication yes`

### vscode安装ssh

![{0FE89677-6178-495F-BB69-24CB658E4BFA}.png](https://github.com/Davereminisce/image/blob/master/%7B0FE89677-6178-495F-BB69-24CB658E4BFA%7D.png?raw=true)

vscode安装ssh

![{3FD710BF-4591-415A-A3E8-99618C04F840}.png](https://github.com/Davereminisce/image/blob/master/%7B3FD710BF-4591-415A-A3E8-99618C04F840%7D.png?raw=true)

点击中=中间上面的搜索框 输入>ssh![{7D5636DC-5260-41E8-9044-A1A7037C0392}.png](https://github.com/Davereminisce/image/blob/master/%7B7D5636DC-5260-41E8-9044-A1A7037C0392%7D.png?raw=true)

找到Open SSH Configuration File...并点击

回到vmware打开命令行

![{630773E3-8DCA-4770-95B0-0134600A6D2F}.png](https://github.com/Davereminisce/image/blob/master/%7B630773E3-8DCA-4770-95B0-0134600A6D2F%7D.png?raw=true)

输入hostname获取虚拟机主机名![{6C196501-A4D4-4C69-BC45-F0D55A621919}.png](https://github.com/Davereminisce/image/blob/master/%7B6C196501-A4D4-4C69-BC45-F0D55A621919%7D.png?raw=true)

输入ifconfig获取虚拟机IP地址

回到vscode输入对应要求

![{909F3B14-8106-4289-AA68-AC9E533D447F}.png](https://github.com/Davereminisce/image/blob/master/%7B909F3B14-8106-4289-AA68-AC9E533D447F%7D.png?raw=true)

保存即可

![{EAC5CA91-C048-49C8-9626-C4E8E233C3A2}.png](https://github.com/Davereminisce/image/blob/master/%7BEAC5CA91-C048-49C8-9626-C4E8E233C3A2%7D.png?raw=true)

右键虚拟机然后点击在当前窗口打开

![{D20F6875-8904-49CD-A4CC-A9C7DB01564F}.png](https://github.com/Davereminisce/image/blob/master/%7BD20F6875-8904-49CD-A4CC-A9C7DB01564F%7D.png?raw=true)

选择Linux![{A74625BF-8A95-4512-A076-4E39442871AB}.png](https://github.com/Davereminisce/image/blob/master/%7BA74625BF-8A95-4512-A076-4E39442871AB%7D.png?raw=true)

输入密码即可![{0DC581FF-23BD-4E1F-9AEC-FE4EF3F2516B}.png](https://github.com/Davereminisce/image/blob/master/%7B0DC581FF-23BD-4E1F-9AEC-FE4EF3F2516B%7D.png?raw=true)

此时显示已连接即连接成功

![{AF01A2DB-D35D-4724-92F2-20A55C6374BA}.png](https://github.com/Davereminisce/image/blob/master/%7BAF01A2DB-D35D-4724-92F2-20A55C6374BA%7D.png?raw=true)

此时可打开settings.json

输入

```json
"remote.SSH.remotePlatform": {
    "davereminisce-virtual-machine": "linux"
}
```

使ssh自动连接虚拟机Linux系统![{CA4490C0-4FF7-488E-B979-580C7F916C49}.png](https://github.com/Davereminisce/image/blob/master/%7BCA4490C0-4FF7-488E-B979-580C7F916C49%7D.png?raw=true)

到此VS Code通过remote-ssh远程连接虚拟机配置成功

**（注意：配置ssh时虚拟机需要处于开机状态）**

2024/10/13
