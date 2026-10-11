/**
 * 长音频分片 —— 把 16 kHz 单声道 16 位 PCM wav 切成若干短片段。
 *
 * ## 为什么需要它(实测数据)
 *
 * SenseVoice 只见过 30 秒以内的语音。把整段几十分钟音频一次性喂给
 * sherpa-onnx,**它不报错**,而是静默退化。用同一份 611 秒音频做受控实验
 * (只改输入长度,内容不变):
 *
 * | 输入长度 | 产出字数 | 字/秒 | token 最大时间跳跃 | 峰值内存 |
 * | -------- | -------- | ----- | ------------------ | -------- |
 * | 30 s     | 193      | 6.43  | 0.5 s              | —        |
 * | 60 s     | 329      | 5.48  | 2.6 s              | —        |
 * | 120 s    | 467      | 3.89  | 2.7 s              | 557 MB   |
 * | 240 s    | 212      | 0.88  | 10.5 s             | 1189 MB  |
 * | 360 s    | 159      | 0.44  | 23.2 s             | 2193 MB  |
 * | 611 s    | 173      | 0.28  | 90.7 s             | 5489 MB  |
 *
 * 前 60 秒单独转写有 329 字,比整段 611 秒的 173 字还多 —— **子集产出超过全集**,
 * 这就是静默丢内容的铁证。退化随长度单调加重,内存近似平方增长,4146 秒那集
 * 直接返回空文本(退出码仍是 0)。
 *
 * ## 为什么不用 `ffmpeg -f segment`
 *
 * 固定切点落进词中间的概率很高(语音连续时几乎必然),每切一刀就可能吃掉半个词。
 * 这里先算每 10 ms 一帧的平均幅度,把每个切点在 ±snapSeconds 窗口内吸附到能量
 * 最低处(多数情况是自然停顿)。顺带少起一个进程,也不用解析 ffmpeg 的输出。
 *
 * @module dsh-video-transcribe/segment
 */

import { readFile, writeFile, open } from 'node:fs/promises'
import { join } from 'node:path'

/** 只认这种输入:16 kHz 单声道 16 位 PCM —— 也正是本插件 convert 阶段的产物。 */
const REQUIRED = { audioFormat: 1, channels: 1, bitsPerSample: 16 }

/**
 * 解析 RIFF/WAVE 头,拿到格式与采样数据。
 *
 * 按块遍历而不是假定数据在偏移 44 处:ffmpeg 默认会写一个 `LIST`/`ISFT` 元数据块,
 * 位置不固定。
 *
 * @param buffer - 整个 wav 文件的字节。
 * @returns `{ sampleRate, durationSec, samples }`。
 * @throws 当文件不是 16 位 PCM 单声道 wav 时。
 */
export function parsePcmWav(buffer) {
  if (!Buffer.isBuffer(buffer) || buffer.length < 44) throw new Error('不是有效的 wav(文件太短)')
  if (buffer.toString('latin1', 0, 4) !== 'RIFF' || buffer.toString('latin1', 8, 12) !== 'WAVE') {
    throw new Error('不是 RIFF/WAVE 文件')
  }

  let format
  let dataOffset = -1
  let dataLength = 0
  let offset = 12

  while (offset + 8 <= buffer.length) {
    const id = buffer.toString('latin1', offset, offset + 4)
    const size = buffer.readUInt32LE(offset + 4)
    const body = offset + 8

    if (id === 'fmt ' && body + 16 <= buffer.length) {
      format = {
        audioFormat: buffer.readUInt16LE(body),
        channels: buffer.readUInt16LE(body + 2),
        sampleRate: buffer.readUInt32LE(body + 4),
        bitsPerSample: buffer.readUInt16LE(body + 14),
      }
    } else if (id === 'data') {
      dataOffset = body
      dataLength = Math.max(0, Math.min(size, buffer.length - body))
      break
    }
    offset = body + (size > 0 ? size : 0) + (size % 2)
  }

  if (format === undefined) throw new Error('wav 缺少 fmt 块')
  if (format.audioFormat !== REQUIRED.audioFormat || format.bitsPerSample !== REQUIRED.bitsPerSample) {
    throw new Error(`只支持 16 位 PCM wav(当前 format=${format.audioFormat} bits=${format.bitsPerSample})`)
  }
  if (format.channels !== REQUIRED.channels) throw new Error(`只支持单声道 wav(当前 ${format.channels} 声道)`)
  if (dataOffset < 0 || dataLength < 2) throw new Error('wav 缺少 data 块')

  // 复制一份:原 buffer 的 dataOffset 不保证是 2 的倍数,直接建 Int16Array 视图会抛。
  const count = dataLength >> 1
  const copy = buffer.buffer.slice(
    buffer.byteOffset + dataOffset,
    buffer.byteOffset + dataOffset + count * 2,
  )

  return {
    sampleRate: format.sampleRate,
    durationSec: count / format.sampleRate,
    samples: new Int16Array(copy),
  }
}

/**
 * 只读文件头就能拿到时长 —— 分片决策不需要把几百 MB 的样本读进内存。
 *
 * @param filePath - wav 路径。
 * @returns `{ sampleRate, dataBytes, durationSec }`。
 * @throws 头 64 KB 内找不到 `data` 块时。
 */
export async function readWavInfo(filePath) {
  const handle = await open(filePath, 'r')
  try {
    const header = Buffer.alloc(64 * 1024)
    const { bytesRead } = await handle.read(header, 0, header.length, 0)
    const buffer = header.subarray(0, bytesRead)

    if (buffer.toString('latin1', 0, 4) !== 'RIFF' || buffer.toString('latin1', 8, 12) !== 'WAVE') {
      throw new Error('不是 RIFF/WAVE 文件')
    }

    let sampleRate = 0
    let offset = 12
    while (offset + 8 <= buffer.length) {
      const id = buffer.toString('latin1', offset, offset + 4)
      const size = buffer.readUInt32LE(offset + 4)
      const body = offset + 8

      if (id === 'fmt ' && body + 16 <= buffer.length) {
        sampleRate = buffer.readUInt32LE(body + 4)
      } else if (id === 'data') {
        if (sampleRate <= 0) throw new Error('wav 缺少 fmt 块')
        return { sampleRate, dataBytes: size, durationSec: size / 2 / sampleRate }
      }
      offset = body + (size > 0 ? size : 0) + (size % 2)
    }
    throw new Error('头 64 KB 内没有找到 data 块')
  } finally {
    await handle.close()
  }
}

/**
 * 规划切点。纯函数,给定样本就一定有确定结果。
 *
 * 规则:
 *   - 总长不超过一片 → 不切
 *   - 名义切点每 `targetSec` 一个,再在 ±`snapSec` 内吸附到最安静处
 *   - 任何一段都不短于 `minSec`;尾巴不足 `minSec` 就并进前一片
 *
 * @param options - `{ sampleRate, samples, targetSec, minSec, snapSec }`。
 * @returns 段边界数组 `[{ startSample, endSample }]`。
 */
export function planSegments({ sampleRate, samples, targetSec = 30, minSec = 5, snapSec = 1.5 }) {
  const total = samples.length
  const target = Math.max(1, Math.round(targetSec * sampleRate))
  const min = Math.max(1, Math.round(minSec * sampleRate))

  if (total <= target) return [{ startSample: 0, endSample: total }]

  // 每 10 ms 一帧的平均幅度 —— 用来找停顿。
  const frame = Math.max(1, Math.round(sampleRate / 100))
  const frameCount = Math.floor(total / frame)
  const energy = new Float64Array(frameCount)
  for (let index = 0; index < frameCount; index += 1) {
    const base = index * frame
    let sum = 0
    for (let i = 0; i < frame; i += 1) sum += Math.abs(samples[base + i])
    energy[index] = sum / frame
  }

  /** 把名义切点吸附到窗口内最安静的一帧中点。 */
  const snap = (nominal) => {
    if (snapSec <= 0) return nominal
    const half = Math.round(snapSec * sampleRate)
    const lowFrame = Math.max(0, Math.floor((nominal - half) / frame))
    const highFrame = Math.min(frameCount - 1, Math.floor((nominal + half) / frame))
    if (highFrame <= lowFrame) return nominal

    let quietest = Infinity
    for (let index = lowFrame; index <= highFrame; index += 1) {
      if (energy[index] < quietest) quietest = energy[index]
    }
    if (!Number.isFinite(quietest)) return nominal

    // 把「和最低点一样安静」的帧都算作候选,取中位 —— 落在一段静音的正中,
    // 而不是它最左边那一帧。
    const threshold = quietest * 1.15 + 1e-6
    const candidates = []
    for (let index = lowFrame; index <= highFrame; index += 1) {
      if (energy[index] <= threshold) candidates.push(index)
    }
    const middle = candidates[Math.floor(candidates.length / 2)]
    return middle * frame + Math.floor(frame / 2)
  }

  const cuts = []
  let previous = 0
  for (let nominal = target; nominal + min <= total; nominal += target) {
    const at = snap(nominal)
    if (at - previous < min) continue // 和上一刀挨太近,放弃这一刀
    if (total - at < min) break // 尾巴会太短,直接并进最后一片
    cuts.push(at)
    previous = at
  }

  const bounds = [0, ...cuts, total]
  return bounds.slice(0, -1).map((start, index) => ({ startSample: start, endSample: bounds[index + 1] }))
}

/**
 * 把样本打包成一个规范的 44 字节头的 wav。
 *
 * @param samples - 16 位有符号单声道样本(可以是 subarray 视图)。
 * @param sampleRate - 采样率。
 * @returns wav 字节。
 */
export function buildPcmWav(samples, sampleRate) {
  const dataLength = samples.length * 2
  const header = Buffer.alloc(44)
  header.write('RIFF', 0, 'latin1')
  header.writeUInt32LE(36 + dataLength, 4)
  header.write('WAVE', 8, 'latin1')
  header.write('fmt ', 12, 'latin1')
  header.writeUInt32LE(16, 16) // fmt 块长度
  header.writeUInt16LE(1, 20) // PCM
  header.writeUInt16LE(1, 22) // 单声道
  header.writeUInt32LE(sampleRate, 24)
  header.writeUInt32LE(sampleRate * 2, 28) // 字节率
  header.writeUInt16LE(2, 32) // 块对齐
  header.writeUInt16LE(16, 34) // 位深
  header.write('data', 36, 'latin1')
  header.writeUInt32LE(dataLength, 40)
  return Buffer.concat([header, Buffer.from(samples.buffer, samples.byteOffset, dataLength)])
}

/**
 * 切分一个 16 kHz 单声道 wav,把片段写到 `outDir`。
 *
 * @param inputPath - 源 wav。
 * @param outDir - 片段输出目录(需已存在)。
 * @param options - `{ targetSec, minSec, snapSec, prefix }`。
 * @returns `{ sampleRate, durationSec, segments }`,每段带 `{ index, path, startSec, endSec, durationSec }`。
 */
export async function segmentWav(inputPath, outDir, {
  targetSec = 30,
  minSec = 5,
  snapSec = 1.5,
  prefix = 'chunk',
} = {}) {
  const { sampleRate, durationSec, samples } = parsePcmWav(await readFile(inputPath))
  const ranges = planSegments({ sampleRate, samples, targetSec, minSec, snapSec })

  const segments = []
  for (const [index, range] of ranges.entries()) {
    const slice = samples.subarray(range.startSample, range.endSample)
    const path = join(outDir, `${prefix}-${String(index).padStart(4, '0')}.wav`)
    await writeFile(path, buildPcmWav(slice, sampleRate))
    segments.push({
      index,
      path,
      startSec: Number((range.startSample / sampleRate).toFixed(2)),
      endSec: Number((range.endSample / sampleRate).toFixed(2)),
      durationSec: Number(((range.endSample - range.startSample) / sampleRate).toFixed(2)),
    })
  }

  return { sampleRate, durationSec, segments }
}

/** 中日韩字符与全角标点 —— 拼接片段文稿时判断要不要补空格。 */
const CJK = /[\u2e80-\u303f\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uff00-\uffef]/

/**
 * 把逐片段文稿拼成一篇。
 *
 * 中日韩文字之间**不能**补空格(会污染正文),拉丁文之间必须补。所以看相邻两个
 * 字符:只要其中一个是 CJK,就紧贴;否则补一个空格。
 *
 * @param parts - 各片段文稿(顺序即音频顺序)。
 * @returns 拼接后的文稿。
 */
export function joinSegments(parts) {
  let out = ''
  for (const raw of parts) {
    const part = String(raw ?? '').trim()
    if (part.length === 0) continue
    if (out.length === 0) {
      out = part
      continue
    }
    const previous = out.charAt(out.length - 1)
    const next = part.charAt(0)
    out += CJK.test(previous) || CJK.test(next) ? '' : ' '
    out += part
  }
  return out
}

/** 供自测使用。 */
export const __test = { parsePcmWav, readWavInfo, planSegments, buildPcmWav, segmentWav, joinSegments }
