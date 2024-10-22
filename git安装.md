# git安装&上传至github

## git安装

[知乎教程](https://zhuanlan.zhihu.com/p/242540359)

基本是一直点next

## 上传文件至github

[b站视频](【如何在GitHub仓库上传文件】https://www.bilibili.com/video/BV1fK4y1q7kp?vd_source=64fa735df4e10c3811ddac775f3035f1)

#### 添加文件或文件夹

先创建文件夹（将想要上传的文件放入），右键点击git bash

#### 初始化 Git 仓库

```bash
git init
```

#### 本地分支更改为main分支

有些情况下本地自动是master分支与github上默认分支main相冲突，这里可以更改分支

如果当前不是main分支，可以重命名当前分支为main：

```
git branch -m 当前分支名 main
```

#### 将文件夹添加到暂存区

```bash
git add .
```

#### 提交更改

”  “里面相当于日志

```bash
git commit -m "Add ..... to the repository "
```

#### 添加远程仓库

```bash
git remote add origin "github库的地址"
```

如果想更改现有的链接

```bash
git remote set-url origin "....."
```

#### 远程main分支与本地分支更新

拉取远程更改

```
git pull origin main --allow-unrelated-histories
git add .
git commit -m "Resolve merge conflicts"
```

#### 6.推送到远程仓库

```bash
git push -u origin main 	#这里main是枝名
```

如果没找对main

```bash
git branch					#检查当前分支
git checkout -b main		#可以创建一个并切换到该分支
git push -u origin master	#可以直接切换到存在的那个分支
```



### 每次修改后重新上传的代码

**修改代码**：在本地文件中进行所需的更改。

**暂存更改**：

```bash
git add .
```

这将暂存所有更改的文件。如果只想暂存特定文件，可以使用 `git add <file>`。

**提交更改**：

```bash
git commit -m "Your commit message describing the changes"
```

**拉取远程更改**（确保你的本地库是最新的）：

```bash
git pull origin main
```

**解决可能的合并冲突**（如果有的话），然后再次提交。

**推送更改**：

```bash
git push -u origin main
```
