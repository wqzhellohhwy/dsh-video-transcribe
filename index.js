/**
 * dsh-video-transcribe — 给 DSH 补上「贴链接 → 本地出文稿」的能力。
 *
 * ## 为什么需要它
 *
 * DSH 的网络面只有 `web_search` 与 `web_fetch`,两者都是**文本**通道:没有浏览器、
 * 没有播放器,视频画面与音轨都拿不到。但本机其实已经躺着一条完整的离线语音识别
 * 链路 —— `~/.dsh/speech-to-text/sensevoice` 下的 SenseVoice int8 模型(语音输入
 * 功能用的那份),配套 `sherpa-onnx` 引擎。缺的只是「把音轨取下来」这一段。
 *
 * 本插件把三段接起来,注册成**一个**工具:
 *
 *   1. `yt-dlp`      取音轨(并要求它顺手写一份 info.json,标题/时长从那里来)
 *   2. `ffmpeg`      转 16 kHz 单声道 PCM wav(SenseVoice 的输入规格)
 *   3. `sherpa-onnx-offline` + SenseVoice  本地推理出文稿
 *
 * 全程不联网到任何转写服务、不需要 API Key、视频不出本机。
 *
 * ## 为什么必须分片(这是一条实测结论,不是预防性设计)
 *
 * SenseVoice 只见过 30 秒以内的语音。把整段几十分钟音频一次性交给
 * `sherpa-onnx-offline`,**它不报错,退出码 0,但内容被静默丢掉**。拿同一份
 * 611 秒音频只改输入长度做的受控实验:
 *
 * | 输入长度 | 产出字数 | 字/秒 | 峰值内存 |
 * | -------- | -------- | ----- | -------- |
 * | 30 s     | 193      | 6.43  | —        |
 * | 60 s     | 329      | 5.48  | —        |
 * | 120 s    | 467      | 3.89  | 557 MB   |
 * | 240 s    | 212      | 0.88  | 1189 MB  |
 * | 611 s    | 173      | 0.28  | 5489 MB  |
 *
 * **前 60 秒单独转写的字数比整段 611 秒还多** —— 子集产出超过全集。4146 秒
 * (69 分钟)那集直接返回空文本,而退出码依然是 0。
 *
 * 所以长音频一律先按 `chunkSeconds`(缺省 30 秒)切开,再让**一次** sherpa 调用
 * 吃下全部片段(实测一个输入文件对应一份 JSON 输出,顺序一致),逐片段拼回来。
 * 分片模块见 `./segment.js`,切点会在 ±`snapSeconds` 内吸附到能量最低处,
 * 尽量避免把词切两半。
 *
 * ## 为什么是「作业」而不是「一次调用」
 *
 * 16 分钟音频的转写在 8 线程 CPU 上要跑 6 分钟左右,而工具调用有超时策略
 * (`tool-call-timeout-policy` 会按工具给 `exec.signal` 装截止时间)。所以工具做成
 * **作业制**:`start` 起一个作业并当场等一小会儿(`waitSeconds`),等到了就直接
 * 返回全文;没等到就返回 `jobId`,模型随后用 `status` 取结果。作业状态落在磁盘上
 * (`jobs/<id>/job.json`),DSH 重启后仍然查得到。
 *
 * ## 依赖形态
 *
 * 与 DSH 第三方插件惯例一致:只 import `node:` 内置模块,**不 import 任何
 * `@deepseek-ai/*`**。三个外部可执行文件由 config 指向(`~/.dsh/video-transcribe/bin`
 * 为缺省位置),模型与 tokens 缺省复用本机已有的 SenseVoice 目录。
 *
 * @module dsh-video-transcribe
 */

import { spawn } from 'node:child_process'
import { existsSync } from 'node:fs'
import { mkdir, readFile, readdir, rm, writeFile } from 'node:fs/promises'
import { randomUUID } from 'node:crypto'
import { cpus, homedir } from 'node:os'
import { join } from 'node:path'
import { joinSegments, readWavInfo, segmentWav } from './segment.js'

/** cordis 插件名(与 package.json 的 name 一致)。 */
export const name = 'dsh-video-transcribe'

/** 只有 tools 注册表在位时本插件才有事可做。 */
export const inject = ['tools']

/** SenseVoice 支持的语言;`auto` 交给模型自判。 */
export const LANGUAGES = ['auto', 'zh', 'en', 'ja', 'ko', 'yue']

/** 单次返回给模型的文稿上限(字符);超过则截断并指向文件。 */
const MAX_TEXT_RETURNED = 50_000

/** 子进程 stdout 收集上限:够装下 sherpa 的 JSON(实测约 125 KB),又能挡住病态输出。 */
const MAX_STDOUT_BYTES = 64 * 1024 * 1024

/** 子进程 stderr 只留尾部这么多字节:诊断够用,不撑爆内存。 */
const MAX_STDERR_BYTES = 64 * 1024

/** 各阶段的缺省超时(毫秒)。 */
const TIMEOUTS = {
  download: 30 * 60_000,
  convert: 10 * 60_000,
  transcribe: 30 * 60_000,
}

/** 请求里允许的 action。 */
const ACTIONS = ['start', 'status', 'list', 'cancel']

/* -------------------------------------------------------------------------- */
/* 配置                                                                        */
/* -------------------------------------------------------------------------- */

/** `$DSH_HOME`,缺省 `~/.dsh`。 */
function dshHome() {
  const fromEnv = process.env.DSH_HOME
  return typeof fromEnv === 'string' && fromEnv.length > 0 ? fromEnv : join(homedir(), '.dsh')
}

/** 取一个非空字符串配置,否则回落到缺省值。 */
function str(value, fallback) {
  return typeof value === 'string' && value.trim().length > 0 ? value : fallback
}

/** 取一个正整数配置,否则回落到缺省值。 */
function int(value, fallback) {
  return Number.isFinite(value) && value > 0 ? Math.floor(value) : fallback
}

/** 取一个非负数字配置(允许 0:例如关掉切点吸附),否则回落到缺省值。 */
function num(value, fallback) {
  return Number.isFinite(value) && value >= 0 ? value : fallback
}

/**
 * 把 Loader 行的 config 解析成插件实际使用的值。
 * 每个字段都可省略,所以手写的 `cordis.patch.yml` 行不会产出不可用的配置。
 *
 * @param config - Loader 行的原始 config。
 * @returns 解析后的配置。
 */
export function resolveConfig(config) {
  const raw = config !== null && typeof config === 'object' && !Array.isArray(config) ? config : {}
  const home = dshHome()
  const root = str(raw.root, join(home, 'video-transcribe'))
  const bin = join(root, 'bin')
  const senseVoiceDir = str(
    raw.modelDir,
    join(home, 'speech-to-text', 'sensevoice', 'models', 'sensevoice-onnx'),
  )
  const cores = typeof cpus === 'function' ? cpus().length : 4

  return {
    root,
    jobsDir: join(root, 'jobs'),
    ytDlp: str(raw.ytDlpPath, join(bin, 'yt-dlp.exe')),
    ffmpeg: str(raw.ffmpegPath, join(bin, 'ffmpeg.exe')),
    sherpa: str(raw.sherpaPath, join(bin, 'sherpa-onnx-offline.exe')),
    model: str(raw.modelPath, join(senseVoiceDir, 'model.int8.onnx')),
    tokens: str(raw.tokensPath, join(senseVoiceDir, 'tokens.txt')),
    language: LANGUAGES.includes(raw.language) ? raw.language : 'auto',
    threads: int(raw.threads, Math.max(2, Math.min(8, cores - 1))),
    defaultWaitSeconds: int(raw.waitSeconds, 45),
    maxWaitSeconds: int(raw.maxWaitSeconds, 120),
    maxDurationSec: int(raw.maxDurationSec, 14_400),
    // 分片。chunkSeconds 是实测选出来的:30 秒片长下产出密度最高(6.43 字/秒),
    // 120 秒已经开始掉(3.89),240 秒以上基本是垃圾(0.88)。
    chunkSeconds: int(raw.chunkSeconds, 30),
    minChunkSeconds: int(raw.minChunkSeconds, 5),
    snapSeconds: num(raw.snapSeconds, 1.5),
    // 一次 sherpa 调用最多带多少个片段。100 片 × 30 秒 = 50 分钟音频,命令行长度也很宽裕。
    segmentBatch: int(raw.segmentBatch, 100),
    keepSegments: raw.keepSegments === true,
    // 产出密度低于这个值就告警(正常中文 2~6 字/秒,退化的那批 ≤ 0.88)。
    lowYieldThreshold: num(raw.lowYieldThreshold, 1),
  }
}

/**
 * 检查三个可执行文件与两个模型文件是否都在。
 * 缺什么就报什么 —— 一个说不清缺什么的工具比没有工具更糟。
 *
 * @param cfg - 解析后的配置。
 * @returns 缺失项的说明列表(全在就是空数组)。
 */
export function checkDependencies(cfg) {
  const missing = []
  for (const [label, path] of [
    ['yt-dlp', cfg.ytDlp],
    ['ffmpeg', cfg.ffmpeg],
    ['sherpa-onnx-offline', cfg.sherpa],
    ['SenseVoice 模型', cfg.model],
    ['tokens.txt', cfg.tokens],
  ]) {
    if (!existsSync(path)) missing.push(`${label}: ${path}`)
  }
  return missing
}

/* -------------------------------------------------------------------------- */
/* 子进程                                                                      */
/* -------------------------------------------------------------------------- */

/**
 * 跑一个子进程并收集输出。**参数走数组、不经 shell**,所以 URL 里的引号之类
 * 不会变成命令注入。
 *
 * 不 reject:调用方拿 `code` 判断,失败原因在 `stderr`。
 *
 * @param command - 可执行文件路径。
 * @param args - 参数数组。
 * @param options - `{ cwd, timeoutMs, signal }`。
 * @returns `{ code, stdout, stderr, timedOut, spawnError }`。
 */
export function run(command, args, options = {}) {
  const { cwd, timeoutMs = TIMEOUTS.transcribe, signal } = options
  return new Promise((resolve) => {
    let child
    try {
      child = spawn(command, args, { cwd, windowsHide: true })
    } catch (error) {
      resolve({ code: -1, stdout: '', stderr: String(error?.message ?? error), spawnError: true, timedOut: false })
      return
    }

    const outChunks = []
    const errChunks = []
    let outBytes = 0
    let errBytes = 0
    let timedOut = false
    let settled = false

    const timer = timeoutMs > 0
      ? setTimeout(() => {
          timedOut = true
          try { child.kill() } catch { /* 已经退出 */ }
        }, timeoutMs)
      : undefined

    const onAbort = () => {
      try { child.kill() } catch { /* 已经退出 */ }
    }
    if (signal !== undefined && typeof signal.addEventListener === 'function') {
      if (signal.aborted === true) onAbort()
      else signal.addEventListener('abort', onAbort, { once: true })
    }

    const finish = (payload) => {
      if (settled) return
      settled = true
      if (timer !== undefined) clearTimeout(timer)
      if (signal !== undefined && typeof signal.removeEventListener === 'function') {
        signal.removeEventListener('abort', onAbort)
      }
      resolve(payload)
    }

    child.stdout?.on('data', (chunk) => {
      outBytes += chunk.length
      if (outBytes <= MAX_STDOUT_BYTES) outChunks.push(chunk)
    })
    child.stderr?.on('data', (chunk) => {
      if (errBytes >= MAX_STDERR_BYTES) return
      errBytes += chunk.length
      errChunks.push(chunk)
    })
    child.on('error', (error) => {
      finish({ code: -1, stdout: '', stderr: String(error?.message ?? error), spawnError: true, timedOut: false })
    })
    child.on('close', (code) => {
      finish({
        code: code === null ? -1 : code,
        stdout: Buffer.concat(outChunks).toString('utf8'),
        stderr: Buffer.concat(errChunks).toString('utf8'),
        spawnError: false,
        timedOut,
      })
    })
  })
}

/* -------------------------------------------------------------------------- */
/* 输出解析                                                                    */
/* -------------------------------------------------------------------------- */

/**
 * 把 sherpa-onnx 的 stdout 变成纯文稿。
 *
 * 引擎输出的是一个 JSON 对象(`{"text": "...", "tokens": [...], "words": []}`),
 * 前面可能混着模型加载信息。这里按 `"text"` 字段逐个提取再解转义 —— 比整体
 * `JSON.parse` 稳,因为 stdout 不一定只有那一个 JSON。
 *
 * @param stdout - 子进程的 stdout。
 * @returns 文稿文本(可能为空串)。
 */
export function extractTranscript(stdout) {
  const matches = [...String(stdout).matchAll(/"text"\s*:\s*"((?:[^"\\]|\\.)*)"/g)]
  if (matches.length === 0) return ''
  const parts = matches.map((match) => unescapeJsonString(match[1]))
  return parts.filter((part) => part.length > 0).join(' ').trim()
}

/**
 * 解 JSON 字符串字面量的转义。
 * 先试 `JSON.parse`,失败(源里有裸控制字符之类)再走保守替换。
 *
 * @param raw - 已剥掉外层引号的字符串体。
 * @returns 解码后的文本。
 */
function unescapeJsonString(raw) {
  try {
    return JSON.parse(`"${raw}"`)
  } catch {
    return raw
      .replace(/\\n/g, '\n')
      .replace(/\\r/g, '\r')
      .replace(/\\t/g, '\t')
      .replace(/\\"/g, '"')
      .replace(/\\\//g, '/')
      .replace(/\\\\/g, '\\')
  }
}

/**
 * sherpa 的公共参数。分片与不分片走同一套,结果才有可比性。
 *
 * @param cfg - 解析后的配置。
 * @param language - 期望语言(`auto` 交给模型自判)。
 * @returns 参数数组。
 */
export function sherpaArgs(cfg, language) {
  return [
    `--sense-voice-model=${cfg.model}`,
    `--tokens=${cfg.tokens}`,
    `--sense-voice-language=${LANGUAGES.includes(language) ? language : cfg.language}`,
    '--sense-voice-use-itn=true',
    `--num-threads=${cfg.threads}`,
  ]
}

/**
 * 从 sherpa stdout 里按**输入文件**拆出文稿。
 *
 * 实测 `sherpa-onnx-offline` 对每个输入 wav 打印一行 JSON(`{"lang":…,"text":…}`),
 * 顺序与命令行一致 —— 这是分片方案能「一次调用吃下几十个片段」的前提。
 * 万一输出形态变了(不再是一行一个 JSON),退回整段提取,至少不丢文本。
 *
 * @param stdout - 子进程 stdout。
 * @returns 每个输入文件一段文稿;完全没识别出内容时是空数组。
 */
export function splitTranscripts(stdout) {
  const lines = String(stdout)
    .split('\n')
    .filter((line) => line.trimStart().startsWith('{"lang"'))
  if (lines.length === 0) {
    const whole = extractTranscript(stdout)
    return whole.length > 0 ? [whole] : []
  }
  return lines.map((line) => {
    try {
      const parsed = JSON.parse(line)
      return typeof parsed.text === 'string' ? parsed.text : ''
    } catch {
      return ''
    }
  })
}

/**
 * 转写一批 wav。
 *
 * 一批只付一次模型加载;若回来的文稿段数少于输入文件数(说明有片段没被识别),
 * 就**逐文件重试**把坏片段隔离出来 —— 一个坏片段既不该毁掉整篇文稿,也不该被
 * 静默丢掉。
 *
 * @param cfg - 解析后的配置。
 * @param files - wav 路径列表。
 * @param options - `{ language, signal, logger }`。
 * @returns `{ ok, texts, perFile, raws, calls, elapsedMs, code?, message? }`。
 */
export async function transcribeFiles(cfg, files, { language, signal, logger } = {}) {
  const texts = []
  const perFile = []
  const raws = []
  const started = Date.now()
  let calls = 0
  const batch = Math.max(1, cfg.segmentBatch)

  const push = (text) => {
    texts.push(text)
    perFile.push(text.length)
  }

  /** 转写单个文件。 */
  const one = async (file) => {
    calls += 1
    const result = await run(cfg.sherpa, [...sherpaArgs(cfg, language), file], {
      timeoutMs: TIMEOUTS.transcribe,
      signal,
    })
    if (result.timedOut) return { timedOut: true }
    raws.push(result.stdout)
    return { text: splitTranscripts(result.stdout).join(' ') }
  }

  for (let start = 0; start < files.length; start += batch) {
    const group = files.slice(start, start + batch)

    if (group.length === 1) {
      const single = await one(group[0])
      if (single.timedOut === true) return { ok: false, code: 'transcribe_timeout', message: '语音识别超时' }
      push(single.text)
      continue
    }

    calls += 1
    const result = await run(cfg.sherpa, [...sherpaArgs(cfg, language), ...group], {
      timeoutMs: TIMEOUTS.transcribe,
      signal,
    })
    if (result.timedOut) return { ok: false, code: 'transcribe_timeout', message: '语音识别超时' }
    raws.push(result.stdout)

    const parts = splitTranscripts(result.stdout)
    if (parts.length === group.length) {
      for (const text of parts) push(text)
      continue
    }

    logger?.warn?.(`一批 ${group.length} 个片段只回来 ${parts.length} 段文稿,改为逐片段重试`)
    for (const file of group) {
      const single = await one(file)
      if (single.timedOut === true) return { ok: false, code: 'transcribe_timeout', message: '语音识别超时' }
      push(single.text)
    }
  }

  return { ok: true, texts, perFile, raws, calls, elapsedMs: Date.now() - started }
}

/**
 * 事后体检:找出「看起来还是丢了内容」的信号。
 *
 * 这些只是**告警**,绝不把作业判失败 —— 静默丢内容才是要防的事,而误报一个
 * 告警的代价远小于让模型把残缺文稿当成完整文稿用。
 *
 * @param options - `{ text, durationSec, segments, threshold }`。
 * @returns 告警文案列表(没有异常就是空数组)。
 */
export function yieldWarnings({ text, durationSec, segments, threshold }) {
  const warnings = []
  if (Number.isFinite(durationSec) && durationSec >= 60 && threshold > 0) {
    const rate = text.length / durationSec
    if (rate < threshold) {
      warnings.push(
        `产出密度只有 ${rate.toFixed(2)} 字/秒(低于 ${threshold};正常中文约 2~6,英文更高),` +
        '可能有内容没被识别出来,建议抽查后再引用',
      )
    }
  }
  if (Array.isArray(segments)) {
    const silent = segments.filter((segment) => segment.durationSec >= 10 && (segment.chars ?? 0) === 0)
    if (silent.length > 0) {
      const ids = silent.slice(0, 8).map((segment) => segment.index).join(', ')
      warnings.push(`${silent.length} 个片段完全没有文字(片段号 ${ids}${silent.length > 8 ? ' …' : ''})`)
    }
  }
  return warnings
}

/** 中日韩字符(含假名)—— 用来判断一份文稿该不该报「词数」。 */
const CJK_CHAR = /[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]/

/**
 * 词数。只有拉丁文为主的文稿,「词数」才有意义。
 *
 * 中文没有词间空格,按空白切会把整段算成几个词 —— 实测一个 69 分钟视频被报成
 * 「24 词」,纯属误导。所以中文一律不给词数(返回 `undefined`,由 `compact` 剥掉,
 * 渲染时也不显示这一项)。
 *
 * @param text - 文稿。
 * @returns 词数;中文或不适用时是 `undefined`。
 */
export function countWords(text) {
  const value = String(text ?? '')
  if (value.length === 0) return undefined
  let cjk = 0
  for (const char of value) {
    if (CJK_CHAR.test(char)) cjk += 1
  }
  if (cjk / value.length > 0.3) return undefined
  const words = value.split(/\s+/).filter((word) => word.length > 0).length
  return words > 0 ? words : undefined
}

/** URL 必须是 http(s):挡掉 `file:` 之类的本地读取面。 */
export function isHttpUrl(value) {
  if (typeof value !== 'string' || value.length === 0) return false
  try {
    const parsed = new URL(value)
    return parsed.protocol === 'http:' || parsed.protocol === 'https:'
  } catch {
    return false
  }
}

/** 生成按时间可排序的作业 id。 */
function newJobId() {
  const now = new Date()
  const pad = (n) => String(n).padStart(2, '0')
  const stamp = `${now.getFullYear()}${pad(now.getMonth() + 1)}${pad(now.getDate())}` +
    `-${pad(now.getHours())}${pad(now.getMinutes())}${pad(now.getSeconds())}`
  return `${stamp}-${randomUUID().slice(0, 6)}`
}

/* -------------------------------------------------------------------------- */
/* 作业存储                                                                    */
/* -------------------------------------------------------------------------- */

/**
 * 建作业仓库:内存里是权威,磁盘上是备份。
 * 磁盘那份让 DSH 重启后 `list` / `status` 仍然可用。
 *
 * @param cfg - 解析后的配置。
 * @param logger - 诊断出口。
 * @returns 仓库句柄。
 */
export function createRegistry(cfg, logger) {
  /** @type {Map<string, object>} */
  const jobs = new Map()

  /** 串行队列:一次只跑一个作业 —— sherpa 推理峰值内存按 GB 计,不允许并发。 */
  let queueTail = Promise.resolve()

  /** 当前在跑的子进程(取消用)。 */
  let current = null

  const jobDir = (id) => join(cfg.jobsDir, id)
  const jobFile = (id) => join(jobDir(id), 'job.json')

  /** 落盘作业快照;失败只告警,不影响作业本身。 */
  async function persist(job) {
    try {
      await mkdir(jobDir(job.id), { recursive: true })
      await writeFile(jobFile(job.id), `${JSON.stringify(job, null, 2)}\n`, 'utf8')
    } catch (error) {
      logger?.warn?.(`作业 ${job.id} 状态写入失败:${String(error?.message ?? error)}`)
    }
  }

  /** 从磁盘读一个作业(内存里没有时)。 */
  async function load(id) {
    try {
      const text = await readFile(jobFile(id), 'utf8')
      const parsed = JSON.parse(text)
      jobs.set(id, parsed)
      return parsed
    } catch {
      return undefined
    }
  }

  /** 取作业:先内存后磁盘。 */
  async function get(id) {
    return jobs.get(id) ?? await load(id)
  }

  /** 列出作业,新的在前。 */
  async function list(limit = 20) {
    try {
      const entries = await readdir(cfg.jobsDir, { withFileTypes: true })
      for (const entry of entries) {
        if (entry.isDirectory() && !jobs.has(entry.name)) await load(entry.name)
      }
    } catch {
      // 目录还不存在 = 还没有任何作业
    }
    return [...jobs.values()]
      .sort((a, b) => String(b.id).localeCompare(String(a.id)))
      .slice(0, limit)
  }

  /* ---- 流水线 ---------------------------------------------------------- */

  /**
   * 跑完一个作业的全部三个阶段。
   * 每个阶段前更新 `phase` 并落盘,所以 `status` 任何时候都能答「卡在哪一步」。
   *
   * @param job - 作业记录(就地修改)。
   * @param signal - 宿主的取消信号。
   */
  async function execute(job, signal) {
    const started = Date.now()
    job.status = 'running'
    job.startedAt = new Date().toISOString()
    const dir = jobDir(job.id)

    const step = async (phase, fn) => {
      job.phase = phase
      job.updatedAt = new Date().toISOString()
      await persist(job)
      const result = await fn()
      if (result !== undefined && typeof result === 'object' && result.ok === false) {
        throw Object.assign(new Error(result.message), { code: result.code })
      }
      return result
    }

    try {
      await step('download', async () => {
        const result = await run(cfg.ytDlp, [
          '--no-warnings',
          '--no-playlist',
          '--write-info-json',
          '-f', 'bestaudio/best',
          '-o', join(dir, 'source.%(ext)s'),
          job.url,
        ], { timeoutMs: TIMEOUTS.download, signal })
        if (result.timedOut) return { ok: false, code: 'download_timeout', message: `下载超时(${TIMEOUTS.download / 60_000} 分钟)` }
        if (result.code !== 0) return { ok: false, code: 'download_failed', message: `yt-dlp 退出码 ${result.code}:${tail(result.stderr)}` }

        const files = await readdir(dir)
        const audio = files.find((f) => f.startsWith('source.') && !f.endsWith('.info.json'))
        if (audio === undefined) return { ok: false, code: 'download_empty', message: 'yt-dlp 成功退出但没有产出音轨文件' }
        job.audioFile = join(dir, audio)

        const infoName = files.find((f) => f.endsWith('.info.json'))
        if (infoName !== undefined) {
          try {
            const info = JSON.parse(await readFile(join(dir, infoName), 'utf8'))
            job.meta = {
              title: typeof info.title === 'string' ? info.title : undefined,
              uploader: typeof info.uploader === 'string' ? info.uploader : undefined,
              durationSec: Number.isFinite(info.duration) ? Math.round(info.duration) : undefined,
              extractor: typeof info.extractor_key === 'string' ? info.extractor_key : undefined,
            }
          } catch {
            job.meta = {}
          }
        }
        if (Number.isFinite(job.meta?.durationSec) && job.meta.durationSec > cfg.maxDurationSec) {
          return {
            ok: false,
            code: 'too_long',
            message: `视频 ${job.meta.durationSec}s 超过上限 ${cfg.maxDurationSec}s(可在 config 调 maxDurationSec)`,
          }
        }
        return { ok: true }
      })

      await step('convert', async () => {
        job.wavFile = join(dir, 'audio16k.wav')
        const result = await run(cfg.ffmpeg, [
          '-hide_banner', '-loglevel', 'error', '-y',
          '-i', job.audioFile,
          '-ac', '1', '-ar', '16000', '-c:a', 'pcm_s16le',
          job.wavFile,
        ], { timeoutMs: TIMEOUTS.convert, signal })
        if (result.timedOut) return { ok: false, code: 'convert_timeout', message: 'ffmpeg 转码超时' }
        if (result.code !== 0) return { ok: false, code: 'convert_failed', message: `ffmpeg 退出码 ${result.code}:${tail(result.stderr)}` }
        return { ok: true }
      })

      // 决定要不要分片。取「wav 头算出的时长」与「info.json 里的时长」中较大的那个:
      // 任一来源算小了都不会让长音频漏过分片,而来源算大了最坏也只是多切几片。
      let durationSec = Number.isFinite(job.meta?.durationSec) ? job.meta.durationSec : 0
      try {
        const info = await readWavInfo(job.wavFile)
        if (Number.isFinite(info.durationSec) && info.durationSec > 0) {
          durationSec = Math.max(durationSec, info.durationSec)
        }
      } catch (error) {
        logger?.warn?.(`作业 ${job.id} 读 wav 头失败,改用时长字段决策:${String(error?.message ?? error)}`)
      }

      /** 真正要送去识别的 wav 列表:不分片时就只有整段那一个。 */
      let wavFiles = [job.wavFile]

      if (durationSec > cfg.chunkSeconds) {
        await step('segment', async () => {
          try {
            const chunksDir = join(dir, 'chunks')
            await mkdir(chunksDir, { recursive: true })
            const result = await segmentWav(job.wavFile, chunksDir, {
              targetSec: cfg.chunkSeconds,
              minSec: cfg.minChunkSeconds,
              snapSec: cfg.snapSeconds,
              prefix: 'chunk',
            })
            wavFiles = result.segments.map((segment) => segment.path)
            job.chunked = true
            job.chunkSeconds = cfg.chunkSeconds
            job.segmentPlan = result.segments.map((segment) => ({
              index: segment.index,
              startSec: segment.startSec,
              endSec: segment.endSec,
              durationSec: segment.durationSec,
            }))
            // 分片时以真正解析出来的音频长度为准 —— 告警里的「字/秒」要准。
            if (Number.isFinite(result.durationSec) && result.durationSec > 0) durationSec = result.durationSec
            return { ok: true }
          } catch (error) {
            // 不回落成单遍:单遍在这条链路上就是「静默丢内容」,宁可响亮地失败。
            return { ok: false, code: 'segment_failed', message: `音频分片失败:${String(error?.message ?? error)}` }
          }
        })
      }

      await step('transcribe', async () => {
        const outcome = await transcribeFiles(cfg, wavFiles, { language: job.language, signal, logger })
        if (outcome.ok !== true) return { ok: false, code: outcome.code, message: outcome.message }

        const text = joinSegments(outcome.texts)
        if (text.length === 0) {
          return { ok: false, code: 'transcribe_empty', message: '识别没有产出文本(音轨可能是纯音乐或静音)' }
        }

        job.text = text
        job.charCount = text.length
        job.wordCount = countWords(text)
        job.textPath = join(dir, 'transcript.txt')
        await writeFile(job.textPath, `${text}\n`, 'utf8')
        job.transcribeCalls = outcome.calls

        try {
          await writeFile(join(dir, 'transcript_raw.txt'), outcome.raws.join('\n'), 'utf8')
        } catch {
          // 原始输出是可选的诊断件
        }

        if (job.segmentPlan !== undefined) {
          job.segments = job.segmentPlan.map((segment, index) => ({
            ...segment,
            chars: outcome.perFile[index] ?? 0,
          }))
          job.segmentsPath = join(dir, 'segments.json')
          await writeFile(job.segmentsPath, `${JSON.stringify(job.segments, null, 2)}\n`, 'utf8')
        }

        job.warnings = yieldWarnings({
          text,
          durationSec,
          segments: job.segments,
          threshold: cfg.lowYieldThreshold,
        })
        return { ok: true }
      })

      // 片段文件对完成后的作业没有用了(文稿与每片字数都已落盘)。
      if (job.chunked === true && cfg.keepSegments !== true) {
        try {
          await rm(join(dir, 'chunks'), { recursive: true, force: true })
        } catch {
          // 删不掉就留着,不影响结果
        }
      }

      job.status = 'done'
      job.phase = 'done'
      job.elapsedMs = Date.now() - started
      job.updatedAt = new Date().toISOString()
      await persist(job)
      logger?.info?.(
        `作业 ${job.id} 完成:${job.charCount} 字符 / 用时 ${Math.round(job.elapsedMs / 1000)}s` +
        (job.chunked === true ? ` / ${job.segments?.length ?? 0} 片` : '') +
        (job.warnings?.length > 0 ? ` / ⚠️ ${job.warnings.length} 条告警` : ''),
      )
    } catch (error) {
      job.status = 'failed'
      job.phase = 'failed'
      job.error = { code: error?.code ?? 'pipeline_failed', message: String(error?.message ?? error) }
      job.elapsedMs = Date.now() - started
      job.updatedAt = new Date().toISOString()
      await persist(job)
      logger?.warn?.(`作业 ${job.id} 失败:${job.error.message}`)
    } finally {
      current = null
    }
  }

  /**
   * 建一个作业并入队。立即返回,不等流水线。
   *
   * @param options - `{ url, language, label, signal }`。
   * @returns 新建的作业记录。
   */
  async function start({ url, language, label, signal }) {
    const id = newJobId()
    const job = {
      id,
      url,
      label: typeof label === 'string' && label.length > 0 ? label : undefined,
      language: LANGUAGES.includes(language) ? language : cfg.language,
      status: 'queued',
      phase: 'queued',
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
      dir: jobDir(id),
    }
    jobs.set(id, job)
    await mkdir(jobDir(id), { recursive: true })
    await persist(job)

    queueTail = queueTail.then(
      () => {
        current = job
        return execute(job, signal)
      },
      () => {
        current = job
        return execute(job, signal)
      },
    )
    return job
  }

  /** 取消运行中的作业。排队中的作业会被标记为取消,轮到它时直接跳过。 */
  function cancel(id) {
    const job = jobs.get(id)
    if (job === undefined) return false
    if (job.status === 'done' || job.status === 'failed') return false
    job.status = 'cancelled'
    job.phase = 'cancelled'
    job.error = { code: 'cancelled', message: '作业已取消' }
    job.updatedAt = new Date().toISOString()
    void persist(job)
    if (current?.id === id) {
      try { current = null } catch { /* 忽略 */ }
    }
    return true
  }

  return { start, get, list, cancel, jobs }
}

/** 取 stderr 末尾若干字符用于报错(整段太长)。 */
function tail(text, max = 400) {
  const s = String(text ?? '').trim()
  return s.length <= max ? s : `…${s.slice(-max)}`
}

/**
 * 递归剥掉 `undefined` 字段。
 *
 * 宿主要求工具返回值是**无损 JSON**:一个值为 `undefined` 的键会让整次调用被判
 * `value is not lossless JSON` 而拒收 —— 作业记录里 `label` / `meta` / `charCount`
 * 这类字段天然可能缺席,所以出口统一过这里,而不是在每个构造点记得删。
 *
 * @param value - 任意返回值。
 * @returns 去掉 undefined 的等价结构。
 */
export function compact(value) {
  if (Array.isArray(value)) return value.map((item) => compact(item))
  if (value !== null && typeof value === 'object') {
    const out = {}
    for (const [key, item] of Object.entries(value)) {
      if (item === undefined) continue
      out[key] = compact(item)
    }
    return out
  }
  return value
}

/* -------------------------------------------------------------------------- */
/* 工具的对外形态                                                              */
/* -------------------------------------------------------------------------- */

/**
 * 把作业记录变成给模型看的文本。
 * 全函数:value 是什么形状都不抛。
 *
 * @param value - execute 的返回值。
 * @returns 一条工具结果文本。
 */
export function formatResult(value) {
  if (value === null || typeof value !== 'object') return 'video_transcribe:空结果'
  const v = /** @type {Record<string, any>} */ (value)

  if (typeof v.error === 'string') {
    return `❌ video_transcribe 失败(${v.code ?? 'error'}):${v.error}`
  }

  if (v.action === 'list') {
    const jobs = Array.isArray(v.jobs) ? v.jobs : []
    if (jobs.length === 0) return '还没有任何转写作业。'
    const lines = jobs.map((job) => {
      const meta = job.meta?.title !== undefined ? ` · ${job.meta.title}` : ''
      return `- ${job.id} [${job.status}/${job.phase}]${meta}${job.charCount !== undefined ? ` · ${job.charCount} 字符` : ''}`
    })
    return `共 ${jobs.length} 个作业(新→旧):\n${lines.join('\n')}`
  }

  if (v.action === 'cancel') {
    return v.cancelled === true
      ? `已取消作业 ${v.jobId}。`
      : `作业 ${v.jobId} 不在可取消状态(可能已完成)。`
  }

  const head = []

  if (v.status === 'done') {
    const meta = v.meta ?? {}
    head.push('✅ 转写完成')
    if (meta.title !== undefined) head.push(`标题:${meta.title}`)
    if (meta.uploader !== undefined) head.push(`作者:${meta.uploader}`)
    if (Number.isFinite(meta.durationSec)) head.push(`时长:${meta.durationSec} 秒`)
    head.push(`语言参数:${v.language} · 用时 ${Math.round((v.elapsedMs ?? 0) / 1000)} 秒`)
    head.push(
      v.wordCount === undefined
        ? `字符数:${v.charCount ?? 0}`
        : `字符数:${v.charCount ?? 0} · 词数:${v.wordCount}`,
    )
    if (v.chunked === true) {
      head.push(
        `分片转写:${v.segmentCount ?? 0} 片 × 约 ${v.chunkSeconds ?? 0} 秒` +
        `(长音频整段交给引擎会静默丢内容,所以必须分片)`,
      )
    }
    if (typeof v.textPath === 'string') head.push(`全文文件:${v.textPath}`)
    if (typeof v.segmentsPath === 'string') head.push(`每片字数:${v.segmentsPath}`)
    if (Array.isArray(v.warnings) && v.warnings.length > 0) {
      head.push('')
      for (const warning of v.warnings) head.push(`⚠️ ${warning}`)
    }
    const text = typeof v.text === 'string' ? v.text : ''
    const shown = text.length > MAX_TEXT_RETURNED
      ? `${text.slice(0, MAX_TEXT_RETURNED)}\n\n…(已截断,全文见 ${v.textPath})`
      : text
    head.push('', '--- 文稿 ---', shown)
    return head.join('\n')
  }

  if (v.status === 'failed') {
    head.push('❌ 转写失败')
    head.push(`作业:${v.jobId} · 阶段:${v.phase}`)
    head.push(`原因(${v.error?.code ?? 'unknown'}):${v.error?.message ?? '未知'}`)
    return head.join('\n')
  }

  if (v.status === 'cancelled') {
    return `作业 ${v.jobId} 已取消。`
  }

  // queued / running
  head.push(`⏳ 作业 ${v.jobId} 正在处理`)
  head.push(`状态:${v.status} · 阶段:${v.phase} · 已用 ${Math.round((v.elapsedMs ?? 0) / 1000)} 秒`)
  if (Number.isFinite(v.meta?.durationSec)) head.push(`视频时长:${v.meta.durationSec} 秒`)
  head.push('', '稍后用 action=status 并带上 jobId 取结果(转写通常要跑音频时长的 0.3~0.5 倍)。')
  return head.join('\n')
}

/**
 * 构造 `video_transcribe` 工具定义。
 *
 * @param cfg - 解析后的配置。
 * @param registry - 作业仓库。
 * @param logger - 诊断出口。
 * @returns 可直接交给 `ctx.tools.register` 的定义。
 */
export function createTool(cfg, registry, logger) {
  /** 睡眠。 */
  const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms))

  return {
    name: 'video_transcribe',
    description:
      '把一个视频/音频链接转成文稿:本机 yt-dlp 取音轨 → ffmpeg 转 16k 单声道 → sherpa-onnx(SenseVoice)离线识别。' +
      '全程本地推理,不调用任何云端转写服务,媒体不出本机,无 API 费用。' +
      '支持 B站/YouTube 等 yt-dlp 能处理的站点。' +
      '长音频会自动切成约 30 秒的片段再识别(整段交给引擎会静默丢内容),返回里会写明片数。' +
      'action=start 起作业(带 url;可用 waitSeconds 当场等结果,默认 45 秒),若没跑完返回 jobId;' +
      'action=status 带 jobId 取进度或最终文稿;action=list 列历史作业;action=cancel 取消作业。' +
      '结果里出现 ⚠️ 表示产出密度异常(可能有内容没被识别),引用前务必抽查。' +
      '注意:标点不完整,专有名词(项目名/人名)常有音近错字,引用前需核对。' +
      '得到文稿后由你自己做摘要/翻译/结构化 —— 本工具只负责出文字。',
    parameters: {
      type: 'object',
      required: ['action'],
      additionalProperties: false,
      properties: {
        action: {
          type: 'string',
          enum: ACTIONS,
          description: 'start=新建转写作业;status=查作业;list=列作业;cancel=取消作业',
        },
        url: {
          type: 'string',
          description: 'start 用:视频/音频页面链接(http/https)',
        },
        jobId: {
          type: 'string',
          description: 'status / cancel 用:start 或 list 返回的作业 id',
        },
        language: {
          type: 'string',
          enum: LANGUAGES,
          description: '可选:识别语言;缺省 auto(由模型自判)。中文内容用 zh、英文用 en 更准',
        },
        label: {
          type: 'string',
          description: '可选:给作业起个便于识别的标签(只影响 list 展示)',
        },
        waitSeconds: {
          type: 'integer',
          minimum: 0,
          description: `可选:start 后最多当场等多少秒(默认 ${cfg.defaultWaitSeconds},上限 ${cfg.maxWaitSeconds});0 = 立即返回 jobId`,
        },
      },
    },
    async execute(args, exec) {
      const input = args !== null && typeof args === 'object' ? args : {}
      const action = typeof input.action === 'string' ? input.action : ''
      const signal = exec?.signal

      if (!ACTIONS.includes(action)) {
        return { error: `未知 action:${action || '(空)'}(可选 ${ACTIONS.join(' / ')})`, code: 'bad_action' }
      }

      if (action === 'list') {
        const jobs = await registry.list(20)
        return compact({
          action: 'list',
          jobs: jobs.map((job) => ({
            id: job.id,
            status: job.status,
            phase: job.phase,
            url: job.url,
            label: job.label,
            meta: job.meta,
            charCount: job.charCount,
            textPath: job.textPath,
            createdAt: job.createdAt,
          })),
        })
      }

      if (action === 'cancel') {
        const id = typeof input.jobId === 'string' ? input.jobId : ''
        if (id.length === 0) return { error: 'cancel 需要 jobId', code: 'missing_job_id' }
        return { action: 'cancel', jobId: id, cancelled: registry.cancel(id) }
      }

      if (action === 'status') {
        const id = typeof input.jobId === 'string' ? input.jobId : ''
        if (id.length === 0) return { error: 'status 需要 jobId', code: 'missing_job_id' }
        const job = await registry.get(id)
        if (job === undefined) return { error: `找不到作业 ${id}(用 action=list 看有哪些)`, code: 'job_not_found' }
        return { action: 'status', ...snapshot(job) }
      }

      // ---- action = start ----
      const url = typeof input.url === 'string' ? input.url.trim() : ''
      if (url.length === 0) return { error: 'start 需要 url', code: 'missing_url' }
      if (!isHttpUrl(url)) return { error: `url 必须是 http/https 链接:${url}`, code: 'bad_url' }

      const missing = checkDependencies(cfg)
      if (missing.length > 0) {
        return {
          error: `工具链不完整,缺:${missing.join(';')}。可用 config 覆盖路径,或把文件放到 ${join(cfg.root, 'bin')}`,
          code: 'missing_dependency',
        }
      }

      const job = await registry.start({
        url,
        language: input.language,
        label: input.label,
        signal,
      })

      const waitSeconds = Math.max(
        0,
        Math.min(
          Number.isFinite(input.waitSeconds) ? Math.floor(input.waitSeconds) : cfg.defaultWaitSeconds,
          cfg.maxWaitSeconds,
        ),
      )

      const deadline = Date.now() + waitSeconds * 1000
      while (waitSeconds > 0 && Date.now() < deadline) {
        const current = await registry.get(job.id)
        if (current !== undefined && ['done', 'failed', 'cancelled'].includes(current.status)) break
        await sleep(2000)
      }

      const final = await registry.get(job.id)
      return { action: 'start', ...snapshot(final ?? job) }
    },
    output: {
      schema: { type: 'object', additionalProperties: true, properties: {} },
      render: (_args, value) => [{ type: 'text', text: formatResult(value) }],
    },
  }
}

/**
 * 把作业记录投影成工具返回值。
 * 完成时带上全文;未完时只带状态 —— 避免把半截状态当成结果。
 *
 * @param job - 作业记录。
 * @returns 工具返回值。
 */
function snapshot(job) {
  const base = {
    jobId: job.id,
    status: job.status,
    phase: job.phase,
    url: job.url,
    label: job.label,
    language: job.language,
    meta: job.meta,
    chunked: job.chunked,
    chunkSeconds: job.chunkSeconds,
    segmentCount: Array.isArray(job.segments) ? job.segments.length : job.segmentPlan?.length,
    elapsedMs: job.elapsedMs ?? (job.startedAt !== undefined ? Date.now() - Date.parse(job.startedAt) : 0),
  }
  if (job.status === 'done') {
    return compact({
      ...base,
      text: job.text,
      textPath: job.textPath,
      charCount: job.charCount,
      wordCount: job.wordCount,
      segmentsPath: job.segmentsPath,
      transcribeCalls: job.transcribeCalls,
      warnings: job.warnings,
    })
  }
  if (job.status === 'failed' || job.status === 'cancelled') {
    return compact({ ...base, error: job.error?.message ?? '未知错误', code: job.error?.code })
  }
  return compact(base)
}

/* -------------------------------------------------------------------------- */
/* 插件主体                                                                    */
/* -------------------------------------------------------------------------- */

/**
 * Host 插件主体:注册 `video_transcribe`。
 *
 * @param ctx - Host 上下文。
 * @param config - Loader 行的 config。
 */
export function apply(ctx, config = {}) {
  const cfg = resolveConfig(config)

  /** 诊断出口:宿主 logger 优先,取不到就退回 stderr。 */
  const logger = (() => {
    try {
      const factory = ctx.logger
      if (typeof factory === 'function') {
        const scoped = factory(name)
        if (scoped !== undefined && typeof scoped.info === 'function') return scoped
      }
      if (factory !== undefined && typeof factory.info === 'function') return factory
    } catch {
      // 落到 stderr
    }
    return {
      info: (message) => process.stderr.write(`[${name}] ${message}\n`),
      warn: (message) => process.stderr.write(`[${name}] ${message}\n`),
    }
  })()

  const tools = ctx.tools
  if (tools === undefined || typeof tools.register !== 'function') {
    logger?.warn?.('ctx.tools 不可用,跳过 video_transcribe 注册')
    return
  }

  const registry = createRegistry(cfg, logger)

  try {
    tools.register(createTool(cfg, registry, logger))
  } catch (error) {
    logger?.warn?.(`video_transcribe 注册失败:${String(error?.message ?? error)}`)
    return
  }

  const missing = checkDependencies(cfg)
  if (missing.length > 0) {
    logger?.warn?.(`video_transcribe 已注册,但工具链不完整(调用时会报错):${missing.join(';')}`)
  } else {
    logger?.info?.('video_transcribe 就绪(yt-dlp + ffmpeg + SenseVoice 均在位)')
  }
}

/** 供本包自测使用,不是公开 API —— Loader 只经 {@link apply} 组合本模块。 */
export const __test = {
  resolveConfig,
  run,
  extractTranscript,
  splitTranscripts,
  transcribeFiles,
  yieldWarnings,
  countWords,
  formatResult,
  createTool,
  sherpaArgs,
}
