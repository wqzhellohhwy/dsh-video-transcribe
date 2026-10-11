# dsh-video-transcribe

> 给 DeepSeek Harness 补上「贴一个视频链接 → 本地出文稿」的能力。
> **全程本机推理,不调用任何云端转写服务,媒体不出本机。**

## 为什么需要它

DSH 的网络面只有 `web_search` 和 `web_fetch`,两者都是**文本**通道:没有浏览器、没有播放器,视频的画
面和音轨都拿不到 —— 所以它天然不能「看视频」。

但本机其实已经躺着一条完整的离线语音识别链路:`~/.dsh/speech-to-text/sensevoice` 下的
**SenseVoice int8 模型**(语音输入功能用的那份)配 `sherpa-onnx` 引擎。缺的只是「把音轨取下来」这一段。

本插件把三段接起来,注册成一个工具 `video_transcribe`:

```
yt-dlp                    取音轨(并写一份 info.json,标题/时长从那里来)
  → ffmpeg                转 16 kHz 单声道 PCM wav(SenseVoice 的输入规格)
    → 分片(约 30 秒一片,切点吸附到静音处)
      → sherpa-onnx + SenseVoice   本地推理出文稿(一次调用吃下整组片段)
```

## ⚠️ 长音频必须分片(实测结论,不是预防性设计)

SenseVoice 只见过 30 秒以内的语音。**把整段长音频一次性交给 `sherpa-onnx-offline`,它退出码是 0,
但内容被静默丢掉。** 用同一批真实音频做的受控实验(只改输入长度,内容不变):

| 输入长度 | 产出字数 | 字/秒 | 峰值内存 |
|---|---|---|---|
| 30 s | 193 | 6.43 | — |
| 60 s | 329 | 5.48 | — |
| 120 s | 467 | 3.89 | 557 MB |
| 240 s | 212 | 0.88 | 1,189 MB |
| 360 s | 159 | 0.44 | 2,194 MB |
| 611 s | 173 | 0.28 | 5,490 MB |

**前 60 秒单独转写的字数比整段 611 秒还多** —— 子集产出超过全集。退化随长度单调加重,内存近似
平方增长,到 69 分钟那集直接返回空文本而退出码仍是 0。

修复方式是**先切片再识别**,效果(同一批音频):

| 视频 | 单遍 | 30 秒分片 | 变化 |
|---|---|---|---|
| 10 分 11 秒 | 173 字 / 68.6 s / 5,490 MB | **3,816 字** / 11.2 s / 845 MB | 内容 ×22,内存 ÷6.5 |
| 18 分 09 秒 | 794 字 / 206.8 s / 16,613 MB | **6,386 字** / 18.4 s / 1,393 MB | 内容 ×8.0,内存 ÷12 |
| 69 分 06 秒 | **0 字** / 482 s / **58,620 MB** | **25,500 字** / 69.8 s / 6,927 MB | 从「什么都没有」到 2.5 万字 |

片长扫描(611 秒音频):15 s → 3,932 字;30 s → 3,816;60 s → 3,346;120 s → 2,329;240 s → 1,027。
15~30 秒之间已经到顶(约 6.4 字/秒,就是这段中文口播的真实密度),再往下切收益很小、切点却翻倍,
所以缺省取 **30 秒**。切点会在 ±1.5 秒内吸附到能量最低处,实测比固定切点多留 **3.8%** 的内容。

> 一次 `sherpa-onnx-offline` 调用可以吃**多个** wav(实测一个输入文件对应一份 JSON 输出、顺序一致),
> 所以分片只付一次模型加载 —— 69 分钟视频 139 片只用了 2 次调用。

## 依赖

| 依赖 | 缺省位置 | 说明 |
|---|---|---|
| `yt-dlp.exe` | `<DSH_HOME>/video-transcribe/bin/` | 取音轨 |
| `ffmpeg.exe` | 同上 | 转码 |
| `sherpa-onnx-offline.exe` | 同上 | 推理引擎(sherpa-onnx v1.13.8 官方预编译静态版) |
| `model.int8.onnx` | `<DSH_HOME>/speech-to-text/sensevoice/models/sensevoice-onnx/` | **复用本机已有模型**,无需下载 |
| `tokens.txt` | 同上 | 同上 |

前三个不在位时插件**仍然注册**,但调用会明确告诉你缺什么、以及去哪放 —— 一个说不清缺什么的工具
比没有工具更糟。

## 工具接口

唯一的工具:`video_transcribe`。

| 参数 | 适用 action | 说明 |
|---|---|---|
| `action` | 全部(必填) | `start` / `status` / `list` / `cancel` |
| `url` | `start` | 视频/音频页面链接,仅接受 **http(s)** |
| `jobId` | `status` / `cancel` | `start` 或 `list` 返回的作业 id |
| `language` | `start` | `auto`(缺省)/ `zh` / `en` / `ja` / `ko` / `yue` |
| `label` | `start` | 作业标签,只影响 `list` 展示 |
| `waitSeconds` | `start` | 最多当场等多少秒,缺省 45、上限 120;`0` = 立即返回 |

### 典型用法

```
video_transcribe({ action: "start", url: "https://www.bilibili.com/video/BV...", language: "zh" })
```

`start` 会等一小会儿:等到了就直接返回**全文**,没等到就返回 `jobId` —— 之后:

```
video_transcribe({ action: "status", jobId: "20261008-002519-349225" })
```

`status` 在作业完成时返回完整文稿,所以模型拿到文字后自己做摘要/翻译/结构化即可 —— 本工具只负责出文字。

### 为什么是「作业」而不是「一次调用」

16 分钟音频的转写在 8 线程 CPU 上要跑 3~6 分钟,而 DSH 的工具调用有超时策略
(`tool-call-timeout-policy` 会按工具给 `exec.signal` 装截止时间)。作业制让长任务不撞超时:

- 作业状态落盘在 `<DSH_HOME>/video-transcribe/jobs/<id>/job.json`,**DSH 重启后仍可查**;
- 每个阶段(`download` → `convert` → `segment` → `transcribe`)都会更新 `phase`,所以随时能答
  「卡在哪一步」;
- **作业串行执行**:sherpa 推理峰值内存按 GB 计(单遍 69 分钟音频实测峰值 **58 GB**,分片后 7 GB),
  不允许并发。

## 配置

写在 **profile 的** `cordis.patch.yml`(同 id 按「完整替换语义」覆盖 bundle 层):

```yaml
- id: video-transcribe
  name: dsh-video-transcribe
  config:
    language: zh
    threads: 8
    maxDurationSec: 7200
    chunkSeconds: 30
```

| 字段 | 类型 | 缺省 | 说明 |
|---|---|---|---|
| `root` | string | `<DSH_HOME>/video-transcribe` | 工作根目录 |
| `ytDlpPath` / `ffmpegPath` / `sherpaPath` | string | `<root>/bin/*` | 三个可执行文件 |
| `modelDir` | string | 本机 SenseVoice 目录 | 模型所在目录 |
| `modelPath` / `tokensPath` | string | 同上 | 单独覆盖模型或词表 |
| `language` | string | `auto` | 缺省识别语言 |
| `threads` | number | `min(8, 核数-1)` | 推理线程数 |
| `waitSeconds` | number | `45` | `start` 的缺省等待秒数 |
| `maxWaitSeconds` | number | `120` | 等待上限 |
| `maxDurationSec` | number | `14400` | 视频时长上限,超过直接拒绝 |
| `chunkSeconds` | number | `30` | 分片目标片长。**别调到 120 以上**,实测 240 秒就开始丢内容 |
| `minChunkSeconds` | number | `5` | 片段长度下限;尾巴不足这个长度就并进前一片 |
| `snapSeconds` | number | `1.5` | 切点在多大范围内吸附到最安静处;`0` = 关掉吸附用固定切点 |
| `segmentBatch` | number | `100` | 一次 sherpa 调用最多带几个片段 |
| `keepSegments` | boolean | `false` | 是否保留切片文件(缺省成功后清理) |
| `lowYieldThreshold` | number | `1` | 产出密度低于这个值(字/秒)就在结果里告警 |

## 已知限制

1. **标点不完整**。SenseVoice 用的是 `--sense-voice-use-itn=true`,中文会带逗号句号,但断句不可靠、
   英文基本没标点。要干净的段落划分得由模型后处理。
2. **专有名词常有音近错字**。实测 `llama.cpp` 被识别成 "Lama do c" / "Lama cP"。
   引用术语前务必核对原视频。
3. **不做画面**。这是音频路线:纯画面、无解说的视频它给不出内容。
4. **不绕过登录**。取不到的音轨(需登录/付费/DRM)就取不到,插件不做凭据注入。
5. **串行**。同一时间只跑一个作业,后面的排队。
6. **只出文字,不摘要**。摘要/翻译/结构化交给调用它的模型。
7. **分片是硬要求,不是优化**。如果切片本身失败(比如 wav 头损坏),作业**直接失败**而不是回落成
   单遍 —— 单遍在这条链路上就是「静默丢内容」,宁可响亮地失败。

## 验证

```bash
node selftest.mjs                                    # 39 个用例:纯函数 + 分片 + 注册契约 + 无损 JSON,不联网
node e2e.mjs "https://www.bilibili.com/video/BV..."   # 走真实工具入口跑完一次完整转写
node e2e.mjs                                          # 缺省跑那个 69 分钟视频(修复前产出 0 字)
```

`e2e.mjs` 除了跑完一次真实转写,还会断言:长音频**必须**走分片路径、文稿量达到健康水平、片段文件
已清理、以及**片段列表里塞一个不存在的文件时整篇文稿不会被毁掉、也不会静默少一段**。

`selftest.mjs` 里的注册契约用例是刻意的:宿主缺 `output` 元数据会**拒绝注册**,而单测的假
`register` 不校验,所以那几条断言是自己在把关。

### 实测数据(参考)

| 项 | 值 |
|---|---|
| 视频 | 16 分 41 秒 中文播客(音轨 AAC 44.1 kHz 立体声,8.58 MiB) |
| 下载 + 转码 + 转写总耗时 | **178 秒**(约音频时长的 0.18×) |
| 产出 | 12,192 字符 / 2,396 词 |
| 69 分钟视频(分片后) | 137 秒内出 25,500 字符,峰值内存 6.9 GB |
| 推理引擎 | sherpa-onnx v1.13.8 静态版 + SenseVoice int8,8 线程 |
| 云端调用 | **零** |

## 安装到 profile

插件走标准的 bundle 机制:profile 的 `package.json` 里加依赖 + 加进 `dsh.profile.bundles` ——
本插件自带 `dsh.bundle.patch`,会自己 insert 那一行,不需要手写 `cordis.patch.yml`。

```json
{
  "dependencies": {
    "dsh-video-transcribe": "link:<本仓库绝对路径>"
  },
  "dsh": { "profile": { "bundles": ["@deepseek-ai/dsh-base", "dsh-video-transcribe"] } }
}
```

然后在该 profile 目录跑 `pnpm install`(link 依赖不联网),**重启实例**生效。

> 注意:桌面版(Electron)的 `desktop` profile 由应用独占管理,`dsh plugin --profile desktop` 会被
> CLI 显式拒绝。写 `package.json` 的方式有效,但插件要**重启桌面版**才会上线。

## 设计约定

- 只 import `node:` 内置模块,**不 import 任何 `@deepseek-ai/*`**(第三方插件解析不到,本地副本会
  造成双实例)。
- **工具返回值必须是「无损 JSON」**。宿主会校验,一个值为 `undefined` 的键就让整次调用被判
  `value is not lossless JSON` 而**拒收** —— 注意这是**运行时**校验,假 `register` 的单测覆盖不到
  (本插件第一版就是这样在真机上被打回来的)。所以所有出口统一过 `compact()`,而不是在每个构造
  点记得删字段;`selftest.mjs` 里有一条对**真实磁盘作业**的无损 JSON 断言守着它。
- 子进程参数一律走**数组、不经 shell**,所以 URL 里的引号不会变成命令注入。
- URL 只放行 http(s),挡掉 `file:` 一类本地读取面。
- 任何一步失败都返回**结构化错误码**(`download_failed` / `segment_failed` / `transcribe_empty` …),
  不是一句「失败了」。
- **静默丢内容比报错更糟 —— 这是本插件最重要的一条设计原则。** 具体落成四件事:
  1. 长音频**强制**分片(整段交给引擎就会静默丢内容);
  2. 分片失败时**直接失败,不回落成单遍**(宁可响亮地失败,也不产出残缺文稿);
  3. 事后体检:产出密度低于 `lowYieldThreshold`、或有 ≥10 秒的片段一个字都没出来,就在结果里
     打 ⚠️ —— 只告警不判失败,因为误报一条告警的代价远小于让模型把残缺文稿当完整文稿用;
  4. 每片字数落盘到 `segments.json`,异常一眼可见。
- 分片是**独立模块** `segment.js`,里面全是可单测的纯函数(`planSegments` / `parsePcmWav` /
  `buildPcmWav` / `joinSegments`),不掺任何子进程或网络。

## 开发笔记:改了代码怎么生效

DSH 的 `hmr` 服务能监听插件目录并热重载 Host 模块,但**需要 profile 显式配好 `base` / `root`**:

```yaml
- id: hmr
  name: '@deepseek-ai/dsh-hmr'
  config:
    base: <本插件目录的绝对路径>
    root: ["."]
```

没配的话(`hmr.root` 为空),**整个进程会缓存这份 Host 模块**:改 `index.js` 不起作用,重启才会
重新 import。注意 `link:` 依赖让源码改动**立即可见**,但那只是文件层面的可见 —— 模块缓存归模块缓存,
两者是两回事。

**另外确认**:桌面版(Electron)的 `desktop` profile 由应用独占管理,`dsh plugin --profile desktop`
会被 CLI 显式拒绝;写 `package.json` 的方式有效,但同样要**重启桌面版**才会上线。
